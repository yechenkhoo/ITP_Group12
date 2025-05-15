  const openAccountModalBtn = document.getElementById('openModalBtn');
  const closeAccountModalBtn = document.getElementById('closeModalBtn');
  const Accountmodal = document.getElementById('Accountmodal');

  // Open the modal
  openAccountModalBtn.addEventListener('click', () => {
    Accountmodal.classList.remove('hidden');
  });

  // Close the modal
  closeAccountModalBtn.addEventListener('click', () => {
    Accountmodal.classList.add('hidden');
  });

  // Close modal when clicking outside the modal content
  window.addEventListener('click', (e) => {
    if (e.target == Accountmodal) {
      Accountmodal.classList.add('hidden');
    }
  });