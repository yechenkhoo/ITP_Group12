const compareCheckboxes = document.querySelectorAll('.compare-checkbox');
const compareBtn = document.getElementById('compareBtn');

const deleteForm = document.getElementById('deleteForm');
const deleteBtn = document.getElementById('deleteBtn');
const deleteInput = document.getElementById('deleteVideoIds');

// Function to open a specific tab and update URL
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
    if (event && event.currentTarget) { // Check if event and currentTarget exist (for user clicks)
        event.currentTarget.classList.add("border-blue-500", "text-blue-500");
        event.currentTarget.classList.remove("border-transparent", "hover:text-gray-600", "hover:border-gray-300");
    } else { // For programmatic calls (like initial load)
        const correspondingTabButton = document.querySelector(`[data-tab-target="#${tabId}"]`);
        if (correspondingTabButton) {
            correspondingTabButton.classList.add("border-blue-500", "text-blue-500");
            correspondingTabButton.classList.remove("border-transparent", "hover:text-gray-600", "hover:border-gray-300");
        }
    }

    // Update URL to include the active tab
    const params = new URLSearchParams(window.location.search);
    params.set('tab', tabId);
    // Preserve other parameters like 'view' and 'sort'
    const currentView = params.get('view') || 'list';
    const currentSort = params.get('sort') || '';
    params.set('view', currentView);
    if (currentSort) {
        params.set('sort', currentSort);
    }
    window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`);
}

// Initialize the correct tab on page load
document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const initialTab = params.get('tab') || 'tab1'; // Default to tab1 if no tab parameter
    openTab(null, initialTab); // Pass null for event as it's not a user click
});

// Toggle List/Grid
document.getElementById('toggleViewBtn').addEventListener('click', () => {
    const params = new URLSearchParams(window.location.search);
    const currentView = params.get('view') || 'list';
    const currentTab = params.get('tab') || 'tab1'; // Get the current active tab

    params.set('view', currentView === 'list' ? 'grid' : 'list');
    params.set('tab', currentTab); // Preserve the current active tab

    // preserve sort automatically
    window.location.search = params.toString();
});

function updateCompareDeleteButtonState() {
    const checkedCount = Array.from(compareCheckboxes).filter(cb => cb.checked).length;
    const checked = Array.from(compareCheckboxes).filter(cb => cb.checked).map(cb => cb.dataset.videoId);


    if (checkedCount >= 2) {
        // Enable & turn green
        compareBtn.disabled = false;
        compareBtn.classList.replace('bg-gray-500', 'bg-[#217346]'); // bg-gray → bg-green
    } else {
        // Disable & turn gray
        compareBtn.disabled = true;
        compareBtn.classList.replace('bg-[#217346]', 'bg-gray-500'); // bg-green → bg-gray
    }

    if (checkedCount >= 1) {
        deleteForm.classList.remove('hidden');
        deleteBtn.disabled = false;
        // populate hidden input with comma-separated IDs
        deleteInput.value = checked.join(',');
    } else {
        deleteForm.classList.add('hidden');
        deleteBtn.disabled = true;
    }
}

// Wire up each checkbox
compareCheckboxes.forEach(cb => {
    cb.addEventListener('click', e => e.stopPropagation());
    // wire up change to update buttons
    cb.addEventListener('change', updateCompareDeleteButtonState);
});

// In case any are pre-checked on page load
updateCompareDeleteButtonState();

document.querySelectorAll('tr[data-href]').forEach(row => {
    row.addEventListener('click', () => {
        window.location.href = row.dataset.href;
    });
});