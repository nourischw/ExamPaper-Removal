const sharp = require('sharp');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  
  try {
    const hfToken = process.env.HUGGINGFACE_TOKEN || req.headers['x-hf-token'];
    
    if (!hfToken) {
      return res.status(401).json({ 
        error: 'HUGGINGFACE_TOKEN required',
        hint: 'Get free token at https://huggingface.co/settings/tokens'
      });
    }
    
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const buffer = Buffer.concat(chunks);
    
    const boundaryMatch = (req.headers['content-type'] || '').match(/boundary=(.+)/);
    if (!boundaryMatch) return res.status(400).json({ error: 'No boundary' });
    
    const boundary = boundaryMatch[1];
    const parts = buffer.toString('binary').split('--' + boundary);
    
    let imageBuffer;
    for (const part of parts) {
      if (!part.includes('filename=')) continue;
      const headerEnd = part.indexOf('\r\n\r\n');
      if (headerEnd === -1) continue;
      const contentStart = headerEnd + 4;
      const lastCrlf = part.lastIndexOf('\r\n');
      imageBuffer = Buffer.from(part.slice(contentStart, lastCrlf), 'binary');
      break;
    }
    
    if (!imageBuffer) return res.status(400).json({ error: 'No file uploaded' });
    
    // Resize for API efficiency
    const resized = await sharp(imageBuffer)
      .resize(1024, 1024, { fit: 'inside', withoutEnlargement: true })
      .png()
      .toBuffer();
    
    // Use BiRefNet for salient object detection (foreground/background separation)
    // Then combine with ink detection to identify handwriting
    const hfResponse = await fetch(
      'https://api-inference.huggingface.co/models/ZhengPeng7/BiRefNet',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${hfToken}`,
          'Content-Type': 'image/png'
        },
        body: resized
      }
    );
    
    if (!hfResponse.ok) {
      const errText = await hfResponse.text();
      throw new Error(`HuggingFace error ${hfResponse.status}: ${errText}`);
    }
    
    const maskBuffer = await hfResponse.buffer();
    
    // Get mask dimensions
    const maskMeta = await sharp(maskBuffer).metadata();
    
    // Load mask as grayscale
    const { data: maskGray } = await sharp(maskBuffer)
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });
    
    // Load original as RGB
    const origMeta = await sharp(resized).metadata();
    const origW = origMeta.width;
    const origH = origMeta.height;
    
    const { data: origRGB } = await sharp(resized)
      .raw()
      .toBuffer({ resolveWithObject: true });
    
    // Combine BiRefNet mask with ink analysis
    // BiRefNet outputs near-white for background, dark for foreground objects
    // We want: foreground = handwriting (dark strokes on paper)
    
    // Step 1: Estimate paper background brightness
    const bgValue = 245; // Assume paper is ~245
    
    // Step 2: Create ink mask from original image
    const maskW = maskMeta.width;
    const maskH = maskMeta.height;
    const inkMask = Buffer.alloc(origW * origH).fill(0);
    
    // Scale factors
    const sx = origW / maskW;
    const sy = origH / maskH;
    
    for (let my = 0; my < maskH; my++) {
      for (let mx = 0; mx < maskW; mx++) {
        const maskIdx = my * maskW + mx;
        const maskVal = maskGray[maskIdx];
        
        // Map mask pixel to original pixels
        const ox1 = Math.round(mx * sx);
        const oy1 = Math.round(my * sy);
        const ox2 = Math.round((mx + 1) * sx);
        const oy2 = Math.round((my + 1) * sy);
        
        // Check if this mask region is foreground (dark in mask = foreground)
        const isForeground = maskVal < 200;
        
        if (isForeground) {
          // For each original pixel in this region, check if it's dark (ink)
          for (let oy = oy1; oy < oy2 && oy < origH; oy++) {
            for (let ox = ox1; ox < ox2 && ox < origW; ox++) {
              const oidx = (oy * origW + ox) * 3;
              const r = origRGB[oidx];
              const g = origRGB[oidx + 1];
              const b = origRGB[oidx + 2];
              const brightness = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
              
              // Ink = significantly darker than paper
              const inkDiff = bgValue - brightness;
              if (inkDiff > 45) {
                inkMask[oy * origW + ox] = 1;
              }
            }
          }
        }
      }
    }
    
    // Step 3: Connected component analysis to clean up noise
    const labels = new Int32Array(origW * origH).fill(-1);
    let numLabels = 0;
    const comps = [];
    
    for (let y = 0; y < origH; y++) {
      for (let x = 0; x < origW; x++) {
        const idx = y * origW + x;
        if (!inkMask[idx] || labels[idx] >= 0) continue;
        
        const stack = [idx];
        labels[idx] = numLabels;
        const pixels = [idx];
        
        while (stack.length) {
          const cur = stack.pop();
          const cx = cur % origW;
          const cy = (cur - cx) / origW;
          
          const neighbors = [[cx+1,cy],[cx-1,cy],[cx,cy+1],[cx,cy-1]];
          for (const [nx, ny] of neighbors) {
            if (nx < 0 || nx >= origW || ny < 0 || ny >= origH) continue;
            const nidx = ny * origW + nx;
            if (inkMask[nidx] && labels[nidx] < 0) {
              labels[nidx] = numLabels;
              stack.push(nidx);
              pixels.push(nidx);
            }
          }
        }
        
        if (pixels.length < 8) { labels[idx] = -1; continue; }
        
        let minX = origW, minY = origH, maxX = 0, maxY = 0;
        for (const i of pixels) {
          const px = i % origW;
          const py = (i - px) / origW;
          if (px < minX) minX = px;
          if (py < minY) minY = py;
          if (px > maxX) maxX = px;
          if (py > maxY) maxY = py;
        }
        
        comps.push({
          label: numLabels,
          count: pixels.length,
          minX, minY, maxX, maxY,
          w: maxX - minX + 1,
          h: maxY - minY + 1,
          area: (maxX - minX + 1) * (maxY - minY + 1)
        });
        
        numLabels++;
      }
    }
    
    // Step 4: Filter components - keep only handwriting-sized strokes
    const toErase = new Set();
    
    for (const comp of comps) {
      const { count, w, h, area } = comp;
      
      // Skip noise (very small)
      if (count < 15) continue;
      // Skip very large areas (likely printed text blocks)
      if (area > origW * origH * 0.15) continue;
      
      // Skip thin horizontal lines (printed rules)
      if (h <= 4 && count > 200) continue;
      
      // Keep medium-sized dark components (handwriting strokes)
      // Typical handwriting: 20-200px wide, 10-80px tall
      if (w > 10 && h > 5 && area < 10000) {
        toErase.add(comp.label);
      }
    }
    
    // Step 5: Build final mask
    const finalMask = Buffer.alloc(origW * origH).fill(0);
    for (const comp of comps) {
      if (!toErase.has(comp.label)) continue;
      
      const { minX, minY, maxX, maxY } = comp;
      const pad = 2;
      
      for (let y = Math.max(0, minY - pad); y <= Math.min(origH - 1, maxY + pad); y++) {
        for (let x = Math.max(0, minX - pad); x <= Math.min(origW - 1, maxX + pad); x++) {
          finalMask[y * origW + x] = 1;
        }
      }
    }
    
    // Step 6: Apply mask - erase detected handwriting
    const resultRGB = Buffer.alloc(origW * origH * 3);
    const paperR = 252, paperG = 250, paperB = 246;
    
    for (let i = 0; i < origW * origH; i++) {
      if (finalMask[i]) {
        resultRGB[i * 3] = paperR;
        resultRGB[i * 3 + 1] = paperG;
        resultRGB[i * 3 + 2] = paperB;
      } else {
        resultRGB[i * 3] = origRGB[i * 3];
        resultRGB[i * 3 + 1] = origRGB[i * 3 + 1];
        resultRGB[i * 3 + 2] = origRGB[i * 3 + 2];
      }
    }
    
    const outputBuffer = await sharp(resultRGB, {
      raw: { width: origW, height: origH, channels: 3 }
    }).png().toBuffer();
    
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', 'attachment; filename="cleaned_exam.png"');
    res.send(outputBuffer);
    
  } catch (err) {
    console.error('Error:', err);
    res.status(500).json({ error: err.message || 'Internal error' });
  }
};
