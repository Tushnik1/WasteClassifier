const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const sharp = require("sharp");
const ort = require("onnxruntime-node");

const app = express();
const PORT = 3000;

// Set EJS as the templating engine
app.set("view engine", "ejs");
app.use(express.static("public")); // For serving static files like CSS

// Multer setup for file uploads
const storage = multer.diskStorage({
    destination: "uploads/",
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    },
});
const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        const allowedTypes = ["image/jpeg", "image/png"];
        if (!allowedTypes.includes(file.mimetype)) {
            return cb(new Error("Only JPEG and PNG files are allowed"));
        }
        cb(null, true);
    },
});

// Load the ONNX model
let session;
async function loadModel() {
    const model_path = path.join(__dirname, "models", "model1.onnx")
    session = await ort.InferenceSession.create(model_path);
    console.log(`ONNX model loaded successfully!\nRunning from path: models//model1.onnx`);
}
loadModel();

// Serve the upload page
app.get("/", (req, res) => {
    res.render("index", { label: null });
});

// Handle image upload and classification
app.post("/upload", upload.single("image"), async (req, res) => {
    if (!req.file) {
        return res.status(400).send("No file uploaded.");
    }

    try {
        // Preprocess the image using sharp
        const {data,info} = await sharp(req.file.path)
            .resize(224, 224) // Resize to 224x224
            .toFormat("jpg") // Ensure RGB format
            .raw()
            .toBuffer({ resolveWithObject: true });
        

        // Convert image buffer to a Float32Array for ONNX input
        const floatArray = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) {
            floatArray[i] = data[i] / 255.0; // Normalize pixel values
        }
        // Create ONNX input tensor (1,3,224,224)
        const tensor = new ort.Tensor("float32", floatArray, [1, 3, 224, 224]);
        // Run inference
        const results = await session.run({ input: tensor });
        console.log(results)
        const outputData = results.output.data; // Get prediction
        const predictedClass = outputData.indexOf(Math.max(...outputData)); // Get class with highest probability

        // Remove uploaded file after processing
        // fs.unlinkSync(req.file.path);
        const label1 = ['Recyclable','Organic']

        // Render result
        res.render("index", { label: `${label1[predictedClass]}` });
    } catch (err) {
        console.error("Error processing image:", err);
        res.status(500).send("Error processing image.");
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
