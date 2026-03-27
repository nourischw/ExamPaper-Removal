const sharp = require('sharp');

module.exports = async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    let imageBuffer;
    
    const contentType = req.headers['content-type'] || '';
    
    if (contentType.includes('application/json')) {
      // JSON: { imageUrl: "..." }
      const { imageUrl } = req.body;
      if (!imageUrl) return res.status(400).json({ error: 'No imageUrl provided' });
      
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error('Failed to fetch image');
      imageBuffer = await response.buffer();
    } else {
      // Multipart form data
      const chunks = [];
      for await (const chunk of req) {
        chunks.push(chunk);
      }
      const buffer = Buffer.concat(chunks);
      
      // Simple multipart parser
      const boundaryMatch = contentType.match(/boundary=(.+)/);
      if (!boundaryMatch) {
        return res.status(400).json({ error: 'No boundary found' });
      }
      const boundary = boundaryMatch[1];
      
      const parts = buffer.toString('binary').split('--' + boundary);
      let foundFile = false;
      
      for (const part of parts) {
        if (!part.includes('filename=')) continue;
        
        const headerEnd = part.indexOf('\r\n\r\n');
        if (headerEnd === -1) continue;
        
        const contentStart = headerEnd + 4;
        const lastCrlf = part.lastIndexOf('\r\n');
        const binaryContent = part.slice(contentStart, lastCrlf);
        imageBuffer = Buffer.from(binaryContent, 'binary');
        foundFile = true;
        break;
      }
      
      if (!foundFile) {
        return res.status(400).json({ error: 'No file uploaded' });
      }
    }
    
    // Process
    const resultBuffer = await processExamPaper(imageBuffer);
    
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', 'attachment; filename="cleaned_exam.png"');
    res.send(resultBuffer);
    
  } catch (err) {
    console.error('Error:', err);
    res.status(500).json({ error: err.message || 'Internal error' });
  }
};

async function processExamPaper(inputBuffer) {
  // Load and resize
  let width, height;
  const image = sharp(inputBuffer);
  const metadata = await image.metadata();
  
  const maxDim = 2000;
  if (metadata.width > maxDim || metadata.height > maxDim) {
    const scale = maxDim / Math.max(metadata.width, metadata.height);
    image.resize(Math.round(metadata.width * scale), Math.round(metadata.height * scale));
  }
  
  const { data: grayBuffer, info } = await image
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  width = info.width;
  height = info.height;
  const data = grayBuffer;
  
  // Step 1: Estimate background with sliding window
  const bgMap = Buffer.alloc(width * height);
  const blockSize = 15;
  
  for (let by = 0; by < Math.ceil(height / blockSize); by++) {
    for (let bx = 0; bx < Math.ceil(width / blockSize); bx++) {
      const blockPixels = [];
      const x0 = bx * blockSize;
      const y0 = by * blockSize;
      
      for (let dy = 0; dy < blockSize && y0 + dy < height; dy++) {
        for (let dx = 0; dx < blockSize && x0 + dx < width; dx++) {
          const idx = (y0 + dy) * width + (x0 + dx);
          blockPixels.push(data[idx]);
        }
      }
      blockPixels.sort((a, b) => a - b);
      const bgValue = blockPixels[Math.floor(blockPixels.length * 0.1)] || 245;
      
      for (let dy = 0; dy < blockSize && y0 + dy < height; dy++) {
        for (let dx = 0; dx < blockSize && x0 + dx < width; dx++) {
          const idx = (y0 + dy) * width + (x0 + dx);
          bgMap[idx] = bgValue;
        }
      }
    }
  }
  
  // Step 2: Ink mask
  const inkMask = Buffer.alloc(width * height);
  for (let i = 0; i < width * height; i++) {
    inkMask[i] = (bgMap[i] - data[i]) > 35 ? 1 : 0;
  }
  
  // Step 3: Connected components (4-connected)
  const labels = new Int32Array(width * height).fill(-1);
  let numLabels = 0;
  const comps = [];
  
  for (let i = 0; i < width * height; i++) {
    if (!inkMask[i] || labels[i] >= 0) continue;
    
    const stack = [i];
    const pixels = [];
    
    while (stack.length) {
      const idx = stack.pop();
      if (idx < 0 || idx >= width * height) continue;
      if (labels[idx] >= 0 || !inkMask[idx]) continue;
      
      labels[idx] = numLabels;
      pixels.push(idx);
      
      const x = idx % width;
      const y = Math.floor(idx / width);
      if (x > 0) stack.push(idx - 1);
      if (x < width - 1) stack.push(idx + 1);
      if (y > 0) stack.push(idx - width);
      if (y < height - 1) stack.push(idx + width);
    }
    
    if (pixels.length < 10) continue;
    
    let minX = width, minY = height, maxX = 0, maxY = 0;
    let totalDark = 0;
    
    for (const idx of pixels) {
      const x = idx % width;
      const y = Math.floor(idx / width);
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      totalDark += bgMap[idx] - data[idx];
    }
    
    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    const area = w * h;
    
    comps.push({
      label: numLabels,
      pixels,
      minX, minY, maxX, maxY, w, h, area,
      count: pixels.length,
      fillRatio: pixels.length / area,
      avgDark: totalDark / pixels.length
    });
    
    numLabels++;
  }
  
  // Step 4: Classify answer components
  const toRemove = new Set();
  
  for (const comp of comps) {
    // Skip noise
    if (comp.count < 15) continue;
    // Skip very large areas
    if (comp.area > width * height * 0.2) continue;
    // Skip very thin horizontal lines
    if (comp.h <= 3 && comp.count > 200) continue;
    
    const { fillRatio, avgDark, count } = comp;
    
    // Printed dots: very low fill ratio
    if (fillRatio < 0.1) continue;
    // Solid fills: very high fill ratio
    if (fillRatio > 0.8) continue;
    // Must be dark enough
    if (avgDark < 25) continue;
    // Very small dense strokes that are just dots
    if (fillRatio > 0.55 && count < 60) continue;
    
    toRemove.add(comp.label);
  }
  
  // Step 5: Build and apply erasure mask
  const eraseMask = Buffer.alloc(width * height);
  
  for (const comp of comps) {
    if (!toRemove.has(comp.label)) continue;
    
    const pad = 3;
    const x1 = Math.max(0, comp.minX - pad);
    const y1 = Math.max(0, comp.minY - pad);
    const x2 = Math.min(width - 1, comp.maxX + pad);
    const y2 = Math.min(height - 1, comp.maxY + pad);
    
    for (let y = y1; y <= y2; y++) {
      for (let x = x1; x <= x2; x++) {
        eraseMask[y * width + x] = 1;
      }
    }
  }
  
  // Apply
  const paperColor = 252;
  const result = Buffer.alloc(width * height);
  for (let i = 0; i < width * height; i++) {
    result[i] = eraseMask[i] ? paperColor : data[i];
  }
  
  // Output PNG
  const output = await sharp(result, {
    raw: { width, height, channels: 1 }
  })
    .png()
    .toBuffer();
  
  return output;
}
