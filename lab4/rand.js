const axios = require('axios');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

async function downloadRandomImages(destinationFolder, prefix, numImages) {
    // Ensure the destination folder exists
    if (!fs.existsSync(destinationFolder)) {
        fs.mkdirSync(destinationFolder, { recursive: true });
    }

    // Fetch random images from Unsplash
    for (let i = 0; i < numImages; i++) {
        try {
            const response = await axios.get('https://source.unsplash.com/random', { responseType: 'arraybuffer' });

            // Generate a unique filename prefix using uuid
            const filenamePrefix = `${prefix}_${uuidv4().slice(0, 8)}`;
            const imageExtension = response.headers['content-type'].split('/')[1];

            // Save the image with the prefixed filename
            const imagePath = `${destinationFolder}/${filenamePrefix}.${imageExtension}`;
            fs.writeFileSync(imagePath, response.data);

            console.log(`Downloaded: ${imagePath}`);
        } catch (error) {
            console.error('Error downloading image:', error.message);
        }
    }
}

// Specify the destination folder and prefix
const destinationFolder = 'datasets/training_data';
const filePrefix = '0';

// Specify the number of images to download
const numImages = 50;

// Download random imag
// Download random images
downloadRandomImages(destinationFolder, filePrefix, numImages);