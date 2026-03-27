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
    }
    
    if (!imageBuffer) return res.status(400).json({ error: 'No file uploaded' });
    
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
  
  const maxDim = 2000;
  if (width > maxDim || height > maxDim) {
    const s = maxDim / Math.max(width, height);
    img.resize(Math.round(width * s), Math.round(height * s));
    const newMeta = await img.metadata();
    width = newMeta.width;
    height = newMeta.height;
  }
  
  const { data: gray } = await img.grayscale().raw().toBuffer({ resolveWithObject: true });
  
  // Step 1: Adaptive threshold to get binary image
  // Black = 0, White = 1
  const binary = Buffer.alloc(width * height);
  const blockSize = 25;
  
  for (let by = 0; by < Math.ceil(height / blockSize); by++) {
    for (let bx = 0; bx < Math.ceil(width / blockSize); bx++) {
      const x0 = bx * blockSize;
      const y0 = by * blockSize;
      const x1 = Math.min(x0 + blockSize, width);
      const y1 = Math.min(y0 + blockSize, height);
      
      let sum = 0, count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sum += gray[y * width + x];
          count++;
        }
      }
      const mean = sum / count;
      const thresh = Math.max(60, mean - 8);
      
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          binary[y * width + x] = gray[y * width + x] < thresh ? 1 : 0;
        }
      }
    }
  }
  
  // Step 2: Find all rectangular contours
  const contours = [];
  const visited = Buffer.alloc(width * height).fill(0);
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      if (!binary[idx] || visited[idx]) continue;
      
      // BFS to get connected region
      const stack = [[x, y]];
      const pixels = [];
      visited[idx] = 1;
      
      while (stack.length) {
        const [cx, cy] = stack.pop();
        pixels.push([cx, cy]);
        
        // Check 4 neighbors
        const neighbors = [[cx+1,cy],[cx-1,cy],[cx,cy+1],[cx,cy-1]];
        for (const [nx, ny] of neighbors) {
          if (nx < 1 || nx >= width-1 || ny < 1 || ny >= height-1) continue;
          const nidx = ny * width + nx;
          if (binary[nidx] && !visited[nidx]) {
            visited[nidx] = 1;
            stack.push([nx, ny]);
          }
        }
      }
      
      if (pixels.length < 20) continue;
      
      let minX = width, minY = height, maxX = 0, maxY = 0;
      let sumX = 0, sumY = 0;
      for (const [px, py] of pixels) {
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
        sumX += px; sumY += py;
      }
      
      contours.push({
        pixels,
        minX, minY, maxX, maxY,
        w: maxX - minX + 1,
        h: maxY - minY + 1,
        cx: Math.round(sumX / pixels.length),
        cy: Math.round(sumY / pixels.length),
        count: pixels.length
      });
    }
  }
  
  // Step 3: Classify contours into ANSWER BOXES vs OTHER
  // ANSWER BOXES: small rectangular contours that look like answer squares
  // Strategy: 
  // - Most answer boxes are on the RIGHT side of exam papers
  // - They are small-medium size (not tiny specks, not huge areas)
  // - They are roughly square or slightly tall/wide
  // - They appear in groups (matching section has 4 boxes in a row)
  
  const answerZones = []; // { x1, y1, x2, y2 }
  
  for (const c of contours) {
    const { w, h, minX, maxX, minY, maxY, cx, cy, count } = c;
    const area = w * h;
    
    // Size filter: must be substantial but not huge
    if (area < 400) continue;
    if (area > width * height * 0.12) continue;
    
    const aspect = w / h;
    
    // ANSWER BOX criteria:
    // 1. In the RIGHT half of the page (where answer boxes usually are)
    // 2. Roughly square-ish (aspect between 0.3 and 3.5)
    // 3. Not extremely thin
    if (cx < width * 0.38) continue;
    if (aspect < 0.25 || aspect > 4.0) continue;
    if (h <= 5) continue; // Very thin lines - skip
    
    // Check if this looks like a single answer box (not part of a large text block)
    // Single boxes have high fill ratio (mostly filled)
    // Text characters have lower fill ratio
    const fillRatio = count / area;
    if (fillRatio < 0.15) continue; // Too sparse - likely printed text
    
    // Additional: check if it's isolated (not part of a cluster of many close boxes)
    // Count nearby contours of similar size
    let similarNearby = 0;
    for (const other of contours) {
      if (other === c) continue;
      const dx = Math.abs(cx - other.cx);
      const dy = Math.abs(cy - other.cy);
      // Same row? Similar y?
      if (dx < w * 2 && dy < h * 2) similarNearby++;
    }
    
    // If there are many similar contours nearby, they might be printed text
    // BUT for matching answer boxes (4 boxes), they're all similar nearby
    // So we need to allow groups of up to ~6
    if (similarNearby > 8) continue;
    
    // This looks like an answer box - add to answer zones
    const pad = 5;
    answerZones.push({
      x1: Math.max(0, minX - pad),
      y1: Math.max(0, minY - pad),
      x2: Math.min(width - 1, maxX + pad),
      y2: Math.min(height - 1, maxY + pad)
    });
  }
  
  // Step 4: Also detect horizontal WRITING LINES (for sentence rewrite section)
  // These are thin horizontal strips in the lower portion of the page
  // We find them by looking for runs of dark pixels that form horizontal lines
  
  // Project dark pixels onto Y axis to find strong horizontal lines
  const rowDensity = new Float32Array(height);
  for (let y = 0; y < height; y++) {
    let count = 0;
    for (let x = 0; x < width; x++) {
      if (binary[y * width + x]) count++;
    }
    rowDensity[y] = count;
  }
  
  // Find peaks in row density - these are horizontal lines
  // A writing line creates a peak in density
  const lineRows = [];
  const lineHeight = 6; // Expected height of a writing line in pixels
  
  for (let y = lineHeight; y < height - lineHeight; y++) {
    // Check if this row is a "peak" in density
    // It's a line if it's darker than surroundings
    const localMax = rowDensity[y];
    const threshold = 15; // Minimum dark pixels to count as a line
    
    if (localMax < threshold) continue;
    
    // Check if it's a local maximum
    let isLocalMax = true;
    for (let dy = -lineHeight; dy <= lineHeight; dy++) {
      if (dy === 0) continue;
      if (rowDensity[y + dy] >= localMax) {
        isLocalMax = false;
        break;
      }
    }
    
    if (isLocalMax) {
      lineRows.push(y);
    }
  }
  
  // Group consecutive line rows into line regions
  const writingLines = [];
  if (lineRows.length > 0) {
    let lineStart = lineRows[0];
    let prevRow = lineRows[0];
    
    for (let i = 1; i <= lineRows.length; i++) {
      const row = lineRows[i];
      if (row === undefined || row - prevRow > lineHeight * 1.5) {
        // End of a line region
        const lineY = Math.round((lineStart + prevRow) / 2);
        const lineBottom = prevRow;
        
        // The writing area is ABOVE the line (the space where students write)
        // But we need to find a region that's clearly for writing
        // Typically the line is surrounded by blank space above
        
        // Add the line itself + a bit above as the answer zone
        const zoneTop = Math.max(0, lineStart - 40); // 40px above line for writing space
        const zoneBottom = Math.min(height - 1, lineBottom + 2);
        
        // Only add if this region is in the lower portion of the page (answer area)
        if (lineY > height * 0.35) {
          writingLines.push({ y1: zoneTop, y2: zoneBottom, lineY });
        }
        
        if (row !== undefined) lineStart = row;
      }
      if (row !== undefined) prevRow = row;
    }
  }
  
  // Step 5: Build erasure mask
  const eraseMask = Buffer.alloc(width * height).fill(0);
  
  // Erase answer boxes
  for (const zone of answerZones) {
    for (let y = zone.y1; y <= zone.y2; y++) {
      for (let x = zone.x1; x <= zone.x2; x++) {
        eraseMask[y * width + x] = 1;
      }
    }
  }
  
  // Erase writing lines (fill the entire line strip)
  for (const ln of writingLines) {
    for (let y = ln.y1; y <= ln.y2; y++) {
      for (let x = 0; x < width; x++) {
        // Only erase if there's some content there (not pure background)
        // This prevents erasing blank space unnecessarily
        if (binary[y * width + x]) {
          eraseMask[y * width + x] = 1;
        }
      }
    }
  }
  
  // Step 6: Apply erasure - draw white over masked areas
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
