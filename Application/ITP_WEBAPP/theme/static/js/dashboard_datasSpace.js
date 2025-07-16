const compareCheckboxes = document.querySelectorAll('.compare-checkbox');
const compareBtn        = document.getElementById('compareBtn');

function openTab(event, tabId) {
    var i, tabcontent, tablinks;

    // Hide all tab contents
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.add("hidden");
    }

    // Remove active class from all tab buttons
    tablinks = document.querySelectorAll("[data-tab-target]");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("border-blue-500", "text-blue-500");
        tablinks[i].classList.add("border-transparent", "hover:text-gray-600", "hover:border-gray-300");
    }

    // Show the current tab content
    document.getElementById(tabId).classList.remove("hidden");

    // Add active class to the current tab button
    event.currentTarget.classList.add("border-blue-500", "text-blue-500");
    event.currentTarget.classList.remove("border-transparent", "hover:text-gray-600", "hover:border-gray-300");
}

// Toggle List/Grid—no init needed
document.getElementById('toggleViewBtn').addEventListener('click', () => {
  const params = new URLSearchParams(window.location.search);
  const current = params.get('view') || 'list';
  params.set('view', current === 'list' ? 'grid' : 'list');
  // preserve sort automatically
  window.location.search = params.toString();
});

function updateCompareButtonState() {
  const checkedCount = Array.from(compareCheckboxes).filter(cb => cb.checked).length;

  if (checkedCount >= 2) {
    // Enable & turn green
    compareBtn.disabled = false;
    compareBtn.classList.replace('bg-gray-500', 'bg-[#217346]');   // bg-gray → bg-green
  } else {
    // Disable & turn gray
    compareBtn.disabled = true;
    compareBtn.classList.replace('bg-[#217346]', 'bg-gray-500');   // bg-green → bg-gray
  }
}

// Wire up each checkbox
compareCheckboxes.forEach(cb => {
  cb.addEventListener('change', updateCompareButtonState);
});

// In case any are pre-checked on page load
updateCompareButtonState();