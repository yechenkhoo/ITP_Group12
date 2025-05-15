const Uploadmodal = document.getElementById("Uploadmodal");
const closeUploadModalBtn = document.getElementById("closeModalBtn");
const openUploadModalBtn = document.getElementById("openModalBtn");
const fileDisplayName = document.getElementById("fileDBName");
const fileInput = document.getElementById("videoDBFile");

// Open the modal
openUploadModalBtn.addEventListener('click', () => {
  Uploadmodal.classList.remove('hidden');
});
 
// Close modal when clicking outside the modal content
window.addEventListener('click', (e) => {
  if (e.target == Uploadmodal) {
    // remove all files from the input
    fileInput.value = '';
    fileDisplayName.textContent = 'No file selected';
    Uploadmodal.classList.add('hidden');
  }
});

// get the file name and display it
fileInput.addEventListener('change', (e) => {
  const file = fileInput.files[0];
  fileDisplayName.textContent = file.name;
});