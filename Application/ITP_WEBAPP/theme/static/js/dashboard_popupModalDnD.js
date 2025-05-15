const Uploadmodal = document.getElementById("Uploadmodal");
const closeUploadModalBtn = document.getElementById("closeModalBtn");
const triggerAreas = document.querySelectorAll(".triggerArea");
const modalStudentName = document.getElementById("modalStudentName");
const modalStudentId = document.getElementById("modalStudentId");

// Loop through each trigger area and attach event listeners
triggerAreas.forEach((triggerArea) => {
  // Show styling when dragging over
  triggerArea.addEventListener("dragover", (event) => {
    event.preventDefault(); // Prevent default to allow drop
    triggerArea.classList.add("border-indigo-500"); // Optional styling to show it's ready
  });

  triggerArea.addEventListener("dragenter", (event) => {
    event.preventDefault();
    triggerArea.classList.add("border-indigo-500"); // Optional styling
  });

  triggerArea.addEventListener("dragleave", () => {
    triggerArea.classList.remove("border-indigo-500"); // Remove styling on leave
  });

  triggerArea.addEventListener("drop", (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;

    // Check if any file is dropped
    if (files.length > 0) {
      const file = files[0]; // Only process the first file

      // Get student info from data attributes
      const studentId = triggerArea.getAttribute("data-student-id");
      const studentName = triggerArea.getAttribute("data-student-name");

      // Update modal with student info
      modalStudentName.textContent = studentName;
      modalStudentId.value = studentId;

      // Display the file name in the modal
      const fileName = document.getElementById("fileName");
      fileName.textContent = `${file.name}`;
      console.log(file.name);

      // push file name into the value of the input
      const fileValue = document.getElementById("fileValue");
      fileValue.value = `${file.name}`;

      // Store the file in a hidden input (for form submission)
      const fileInput = document.getElementById("videoFile");
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInput.files = dataTransfer.files;


      // Show modal
      Uploadmodal.classList.remove("hidden");
    }
    triggerArea.classList.remove("border-indigo-500"); // Remove styling on drop
  });
});

// Close modal when clicking outside the modal content
window.addEventListener("click", (e) => {
  if (e.target === Uploadmodal) {
    Uploadmodal.classList.add("hidden");
  }
  
});