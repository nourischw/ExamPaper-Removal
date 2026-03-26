const sharp = require('sharp');
const fetch = require('node-fetch');
const { URL } = require('url');

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
    
    // Handle different input formats
    const contentType = req.headers['content-type'] || '';
    
    if (contentType.includes('application/json')) {
      // JSON: { imageUrl: "..." }
      const { imageUrl } = req.body;
      if (!imageUrl) return res.status(400).json({ error: 'No imageUrl provided' });
      
      const parsed = new URL(imageUrl);
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error('Failed to fetch image');
      imageBuffer = await response.buffer();
    } else {
      // Multipart form data
      const formData = await parseFormData(req);
      const file = formData.files && formData.files[0];
      if (!file) return res.status(400).json({ error: 'No file uploaded' });
      imageBuffer = file.buffer;
    }
    
    // Process the image
    const resultBuffer = await processExamPaper(imageBuffer);
    
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', 'attachment; filename="cleaned_exam.png"');
    res.send(resultBuffer);
    
  } catch (err) {
    console.error('Error:', err.message);
    res.status(500).json({ error: err.message });
  }
};

async function parseFormData(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      const buffer = Buffer.concat(chunks);
      
      // Simple multipart parser
      const boundary = req.headers['content-type'].split('boundary=')[1];
      if (!boundary) {
        resolve({ files: [] });
        return;
      }
      
      const parts = buffer.toString('binary').split('--' + boundary);
      const files = [];
      
      for (const part of parts) {
        if (part.includes('filename=')) {
          const match = part.match(/filename="([^"]+)"/);
          const filename = match ? match[1] : 'upload';
          
          // Find content start (after double CRLF)
          const headerEnd = part.indexOf('\r\n\r\n');
          if (headerEnd === -1) continue;
          
          const contentStart = headerEnd + 4;
          const contentEnd = part.lastIndexOf('\r\n');
          const binaryContent = part.slice(contentStart, contentEnd);
          const contentBuffer = Buffer.from(binaryContent, 'binary');
          
          files.push({ filename, buffer: contentBuffer });
        }
      }
      
      resolve({ files });
    });
    req.on('error', reject);
  });
}

async function processExamPaper(inputBuffer) {
  // Load image with sharp
  const image = sharp(inputBuffer);
  const metadata = await image.metadata();
  let { width, height } = metadata;
  
  // Resize for processing (limit to 2000px max dimension)
  const maxDim = 2000;
  let scale = 1;
  if (width > maxDim || height > maxDim) {
    scale = maxDim / Math.max(width, height);
    width = Math.round(width * scale);
    height = Math.round(height * scale);
    image.resize(width, height);
  }
  
  // Convert to grayscale RGBA
  const { data: rgbaBuffer, info } = await image
    .removeAlpha()
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  width = info.width;
  height = info.height;
  const data = rgbaBuffer;
  
  // Step 1: Estimate background brightness using sliding window
  const bgMap = new Uint8Array(width * height);
  const blockSize = 15;
  
  for (let by = 0; by < Math.ceil(height / blockSize); by++) {
    for (let bx = 0; bx < Math.ceil(width / blockSize); bx++) {
      const blockPixels = [];
      for (let dy = 0; dy < blockSize && (by * blockSize + dy) < height; dy++) {
        for (let dx = 0; dx < blockSize && (bx * blockSize + dx) < width; dx++) {
          const idx = (by * blockSize + dy) * width + (bx * blockSize + dx);
          blockPixels.push(data[idx]);
        }
      }
      blockPixels.sort((a, b) => a - b);
      const bgValue = blockPixels[Math.floor(blockPixels.length * 0.1)] || 245;
      
      for (let dy = 0; dy < blockSize && (by * blockSize + dy) < height; dy++) {
        for (let dx = 0; dx < blockSize && (bx * blockSize + dx) < width; dx++) {
          const idx = (by * blockSize + dy) * width + (bx * blockSize + dx);
          bgMap[idx] = bgValue;
        }
      }
    }
  }
  
  // Step 2: Create ink mask (pixels darker than local background)
  const inkMask = new Uint8Array(width * height);
  
  for (let i = 0; i < width * height; i++) {
    const diff = bgMap[i] - data[i];
    inkMask[i] = diff > 35 ? 255 : 0;
  }
  
  // Step 3: Connected component labeling (4-connected)
  const labels = new Int32Array(width * height).fill(-1);
  let numLabels = 0;
  const compStats = [];
  
  function flood(startIdx) {
    const stack = [startIdx];
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
    
    return pixels;
  }
  
  for (let i = 0; i < width * height; i++) {
    if (inkMask[i] && labels[i] < 0) {
      const pixels = flood(i);
      
      if (pixels.length < 10) continue; // Skip noise
      
      // Compute bounding box
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
      
      const area = (maxX - minX + 1) * (maxY - minY + 1);
      const fillRatio = pixels.length / area;
      const avgDark = totalDark / pixels.length;
      
      compStats.push({
        label: numLabels,
        pixels,
        minX, minY, maxX, maxY,
        w: maxX - minX + 1,
        h: maxY - minY + 1,
        area,
        fillRatio,
        avgDark,
        count: pixels.length
      });
      
      numLabels++;
    }
  }
  
  // Step 4: Classify components
  // Handwritten answers typically have:
  // - Medium fill ratio (not dots, not full fill)
  // - Higher than average darkness
  // - Located in answer regions
  
  // Compute global averages
  const avgFillRatio = compStats.reduce((s, c) => s + c.fillRatio, 0) / (compStats.length || 1);
  const avgDark = compStats.reduce((s, c) => s + c.avgDark, 0) / (compStats.length || 1);
  
  // Identify answer regions
  // Strategy: answers are usually in boxes (small square regions) or on lines
  // We identify them by their position pattern
  
  // Detect answer boxes: clusters of small components in a grid pattern on the right side
  const answerComponents = new Set();
  
  for (const comp of compStats) {
    // Skip very thin (line-like) or very large components
    if (comp.h <= 3 && comp.count > 200) continue;
    if (comp.area > width * height * 0.2) continue;
    if (comp.count < 15) continue;
    
    // Handwritten answer: medium fill ratio, medium darkness
    // Printed dots have very low fill ratio (< 0.15)
    // Solid fills have very high fill ratio (> 0.7)
    const isDenselyFilled = comp.fillRatio > 0.12 && comp.fillRatio < 0.75;
    const isDark = comp.avgDark > 25;
    
    if (isDenselyFilled && isDark) {
      answerComponents.add(comp.label);
    }
  }
  
  // Additional filter: group components that are spatially close
  // Answer boxes tend to cluster in specific regions
  const toRemove = new Set();
  
  for (const compA of compStats) {
    if (!answerComponents.has(compA.label)) continue;
    
    // Check if this component is alone in its region (not part of printed text)
    // Printed text components are usually very small and very dense
    if (compA.fillRatio > 0.5 && compA.count < 50) continue;
    
    toRemove.add(compA.label);
  }
  
  // Step 5: Build erasure mask
  const eraseMask = new Uint8Array(width * height);
  
  for (const comp of compStats) {
    if (toRemove.has(comp.label)) {
      // Expand the bounding box slightly to catch ink edges
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
  }
  
  // Step 6: Apply erasure
  const result = Buffer.alloc(width * height);
  const paperColor = 252; // Near-white paper
  
  for (let i = 0; i < width * height; i++) {
    result[i] = eraseMask[i] ? paperColor : data[i];
  }
  
  // Output as PNG
  const outputBuffer = await sharp(result, {
    raw: { width, height, channels: 1 }
  })
    .png()
    .toBuffer();
  
  return outputBuffer;
}
