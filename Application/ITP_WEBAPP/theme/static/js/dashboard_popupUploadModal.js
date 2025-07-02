const Uploadmodal = document.getElementById("Uploadmodal");
const closeUploadModalBtn = document.getElementById("closeModalBtn");
const openUploadModalBtn = document.getElementById("openModalBtn");

let fileInput;
let fileDisplayNameText;
let fileIcon; // New variable for the SVG icon container
let dropZone;

// Determine which file input, display, icon, and dropzone to use based on the current path
const isStudentDataSpace = window.location.pathname.includes('/home/dataSpace/');

if (isStudentDataSpace) {
    fileInput = document.getElementById("videoDBFile");
    fileDisplayNameText = document.getElementById("fileDBNameText");
    fileIcon = document.getElementById("fileDBIcon"); // Get the SVG icon span
    dropZone = document.getElementById("dropZone");
} else {
    fileInput = document.getElementById("videoFile");
    fileDisplayNameText = document.getElementById("fileNameText");
    fileIcon = document.getElementById("fileIcon"); // Get the SVG icon span
    dropZone = document.getElementById("dropZoneCoach");
}

// Open the modal
openUploadModalBtn.addEventListener('click', () => {
    Uploadmodal.classList.remove('hidden');
});

// Close modal when clicking outside the modal content
window.addEventListener('click', (e) => {
    if (e.target === Uploadmodal) {
        resetFileInput();
        Uploadmodal.classList.add('hidden');
    }
});

// Close modal when clicking the X
closeUploadModalBtn.addEventListener('click', () => {
    resetFileInput();
    Uploadmodal.classList.add('hidden');
});

// Function to reset the file input and display
function resetFileInput() {
    fileInput.value = '';
    fileDisplayNameText.textContent = 'No file selected';
    fileIcon.classList.add('hidden'); // Hide the SVG icon
}

// Handle file selection via browse button
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
        fileDisplayNameText.textContent = file.name;
        fileIcon.classList.remove('hidden'); // Show the SVG icon
    } else {
        fileDisplayNameText.textContent = 'No file selected';
        fileIcon.classList.add('hidden'); // Hide the SVG icon
    }
});

// Drag and Drop functionality
if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault(); // Prevent default to allow drop
        dropZone.classList.add('border-indigo-500', 'bg-indigo-50'); // Add visual feedback
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-indigo-500', 'bg-indigo-50'); // Remove visual feedback
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); // Prevent default behavior (opening file in new tab)
        dropZone.classList.remove('border-indigo-500', 'bg-indigo-50'); // Remove visual feedback

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files; // Assign dropped files to the file input
            fileDisplayNameText.textContent = files[0].name; // Display the name of the first dropped file
            fileIcon.classList.remove('hidden'); // Show the SVG icon for dropped file
        } else {
            fileIcon.classList.add('hidden'); // Hide the SVG if drop results in no file
        }
    });
}