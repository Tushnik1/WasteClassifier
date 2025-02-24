const express = require("express");
const multer = require("multer");
const path = require("path");
const torch = require("@pytorch/torchscript"); // Use PyTorch in Node.js
const fs = require("fs");
const sharp = require("sharp"); // Image processing

const app = express();
const PORT = 3000;

app.set("view engine", "ejs");
app.use(express.static("public"));

const storage = multer.diskStorage({
    destination: "uploads/",
    filename: (req, file, cb) => {
        cb(null, file.fieldname + "-" + Date.now() + path.extname(file.originalname));
    },
});
const upload = multer({ storage });

// Load the TorchScript model
const modelPath = path.join(__dirname, "models", "model.pt");
let model;

async function loadModel() {
    model = await torch.jit.load(modelPath);
    console.log("Model loaded successfully!");
}

loadModel();

// Home route
app.get("/", (req, res) => {
    res.render("index", { label: null });
});

// Image upload and classification
app.post("/upload", upload.single("image"), async (req, res) => {
    if (!req.file) {
        return res.status(400).send("No file uploaded.");
    }

    try {
        // Preprocess image (Resize to 224x224 and convert to Tensor)
        const buffer = await sharp(req.file.path)
            .resize(224, 224)
            .toBuffer();

        let tensor = torch.fromBlob(buffer, [1, 3, 224, 224]); // Shape (Batch, Channels, Height, Width)
        tensor = tensor.div(255.0); // Normalize

        // Run inference
        const output = model.forward(tensor);
        const prediction = output.argmax(1).item(); // Get class index

        // Define class labels
        const classNames = ["Bio-degradable", "Non-biodegradable", "Reusable", "General"]; // Update with actual class names
        const label = classNames[prediction];

        res.render("index", { label });
    } catch (err) {
        console.error(err);
        res.status(500).send("Error processing image.");
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
