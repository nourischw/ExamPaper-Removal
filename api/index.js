const sharp = require('sharp');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  
  try {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const buffer = Buffer.concat(chunks);
    
    const contentType = req.headers['content-type'] || '';
    const boundaryMatch = contentType.match(/boundary=(.+)/);
    
    let imageBuffer;
    if (boundaryMatch) {
      const boundary = boundaryMatch[1];
      const parts = buffer.toString('binary').split('--' + boundary);
      for (const part of parts) {
        if (!part.includes('filename=')) continue;
        const headerEnd = part.indexOf('\r\n\r\n');
        if (headerEnd === -1) continue;
        const contentStart = headerEnd + 4;
        const lastCrlf = part.lastIndexOf('\r\n');
        imageBuffer = Buffer.from(part.slice(contentStart, lastCrlf), 'binary');
        break;
      }
    } else {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    if (!imageBuffer) return res.status(400).json({ error: 'No file data' });
    
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
  const img = sharp(inputBuffer);
  const meta = await img.metadata();
  let width = meta.width;
  let height = meta.height;
  
  const maxDim = 1800;
  if (width > maxDim || height > maxDim) {
    const s = maxDim / Math.max(width, height);
    img.resize(Math.round(width * s), Math.round(height * s));
    const newMeta = await img.metadata();
    width = newMeta.width;
    height = newMeta.height;
  }
  
  const { data: gray } = await img.grayscale().raw().toBuffer({ resolveWithObject: true });
  
  // Estimate paper background (brightest pixels in blocks)
  const bgMap = Buffer.alloc(width * height);
  const blockSize = 20;
  
  for (let by = 0; by < Math.ceil(height / blockSize); by++) {
    for (let bx = 0; bx < Math.ceil(width / blockSize); bx++) {
      const pixels = [];
      for (let dy = 0; dy < blockSize && by * blockSize + dy < height; dy++) {
        for (let dx = 0; dx < blockSize && bx * blockSize + dx < width; dx++) {
          pixels.push(gray[(by * blockSize + dy) * width + (bx * blockSize + dx)]);
        }
      }
      pixels.sort((a, b) => a - b);
      const bg = pixels[Math.floor(pixels.length * 0.05)] || 245;
      for (let dy = 0; dy < blockSize && by * blockSize + dy < height; dy++) {
        for (let dx = 0; dx < blockSize && bx * blockSize + dx < width; dx++) {
          bgMap[(by * blockSize + dy) * width + (bx * blockSize + dx)] = bg;
        }
      }
    }
  }
  
  // Create ink mask: pixels significantly darker than local background
  const inkMask = Buffer.alloc(width * height);
  for (let i = 0; i < width * height; i++) {
    inkMask[i] = (bgMap[i] - gray[i]) > 40 ? 1 : 0;
  }
  
  // Connected components on ink mask (4-connected)
  // inkMask[i] = 1 means this pixel is ink
  const labels = new Int32Array(width * height).fill(-1);
  let numLabels = 0;
  const comps = [];
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!inkMask[idx] || labels[idx] >= 0) continue;
      
      // BFS flood fill
      const stack = [idx];
      labels[idx] = numLabels;
      const pixels = [idx];
      
      while (stack.length) {
        const cur = stack.pop();
        const cx = cur % width;
        const cy = (cur - cx) / width;
        
        if (cx > 0 && inkMask[cur - 1] && labels[cur - 1] < 0) {
          labels[cur - 1] = numLabels;
          stack.push(cur - 1);
          pixels.push(cur - 1);
        }
        if (cx < width - 1 && inkMask[cur + 1] && labels[cur + 1] < 0) {
          labels[cur + 1] = numLabels;
          stack.push(cur + 1);
          pixels.push(cur + 1);
        }
        if (cy > 0 && inkMask[cur - width] && labels[cur - width] < 0) {
          labels[cur - width] = numLabels;
          stack.push(cur - width);
          pixels.push(cur - width);
        }
        if (cy < height - 1 && inkMask[cur + width] && labels[cur + width] < 0) {
          labels[cur + width] = numLabels;
          stack.push(cur + width);
          pixels.push(cur + width);
        }
      }
      
      if (pixels.length < 8) { labels[idx] = -1; continue; }
      
      let minX = width, minY = height, maxX = 0, maxY = 0;
      let sumX = 0, sumY = 0;
      for (const i of pixels) {
        const px = i % width;
        const py = (i - px) / width;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
        sumX += px; sumY += py;
      }
      
      const w = maxX - minX + 1;
      const h = maxY - minY + 1;
      const area = w * h;
      
      comps.push({
        label: numLabels,
        count: pixels.length,
        minX, minY, maxX, maxY, w, h, area,
        cx: Math.round(sumX / pixels.length),
        cy: Math.round(sumY / pixels.length),
        fillRatio: pixels.length / area
      });
      
      numLabels++;
    }
  }
  
  // Classify: which components are ANSWERS?
  // Answers in exam papers are:
  // 1. Answer BOXES: small square filled regions (30-120px), in right half of page
  //    - Matching section: D, A, B, C letters in boxes
  // 2. Answer LINES: handwriting strokes on writing lines (lower part of page)
  //    - Rewrite section: sentences written on lines
  
  const answerBoxes = [];
  const answerStrokes = [];
  
  for (const comp of comps) {
    const { count, w, h, area, fillRatio, cx, cy } = comp;
    
    // Skip huge or tiny
    if (area > width * height * 0.12) continue;
    if (count < 15) continue;
    
    // Skip very thin long lines (printed rules)
    if (h <= 4 && area > 200) continue;
    
    const aspect = w / h;
    
    // CANDIDATE 1: Answer box (matching section letters)
    // Small square-ish filled region in right half of page
    const isInRightHalf = cx > width * 0.4;
    const isSquareish = aspect > 0.25 && aspect < 4.0;
    const isSizedForBox = area > 300 && area < 12000;
    const isSolidFill = fillRatio > 0.30 && fillRatio < 0.80;
    
    if (isInRightHalf && isSquareish && isSizedForBox && isSolidFill) {
      // Make sure this box is relatively isolated (not a printed grid)
      // Count how many other similar boxes are nearby
      let nearbyCount = 0;
      for (const other of comps) {
        if (other.label === comp.label) continue;
        const dx = Math.abs(cx - other.cx);
        const dy = Math.abs(cy - other.cy);
        if (dx < 60 && dy < 60) nearbyCount++;
      }
      // If many neighbors, it's likely a printed character grid - skip
      if (nearbyCount <= 3) {
        answerBoxes.push(comp);
      }
    }
    
    // CANDIDATE 2: Handwriting strokes on answer lines
    // These are in the lower portion of the page (below 35% height)
    // Characteristic: medium density, medium size
    const isInLowerHalf = cy > height * 0.35;
    const isDenselyFilled = fillRatio > 0.35;
    const isMediumSize = count > 20 && area < 8000;
    
    if (isInLowerHalf && isDenselyFilled && isMediumSize) {
      // Additional filter: skip things that look like dots or very small chars
      // and skip things that are too large (likely printed characters)
      if (fillRatio > 0.5 && count < 50) continue; // skip small dense dots
      if (area > 5000 && fillRatio > 0.7) continue; // skip large dense blocks
      answerStrokes.push(comp);
    }
  }
  
  // Build erasure mask
  const eraseMask = Buffer.alloc(width * height).fill(0);
  
  for (const box of answerBoxes) {
    const pad = 3;
    const x1 = Math.max(0, box.minX - pad);
    const y1 = Math.max(0, box.minY - pad);
    const x2 = Math.min(width - 1, box.maxX + pad);
    const y2 = Math.min(height - 1, box.maxY + pad);
    for (let py = y1; py <= y2; py++) {
      for (let px = x1; px <= x2; px++) {
        eraseMask[py * width + px] = 1;
      }
    }
  }
  
  for (const stroke of answerStrokes) {
    const pad = 3;
    const x1 = Math.max(0, stroke.minX - pad);
    const y1 = Math.max(0, stroke.minY - pad);
    const x2 = Math.min(width - 1, stroke.maxX + pad);
    const y2 = Math.min(height - 1, stroke.maxY + pad);
    for (let py = y1; py <= y2; py++) {
      for (let px = x1; px <= x2; px++) {
        eraseMask[py * width + px] = 1;
      }
    }
  }
  
  // Apply erasure
  const result = Buffer.alloc(width * height);
  for (let i = 0; i < width * height; i++) {
    result[i] = eraseMask[i] ? 252 : gray[i];
  }
  
  const output = await sharp(result, {
    raw: { width, height, channels: 1 }
  })
    .png()
    .toBuffer();
  
  return output;
}
