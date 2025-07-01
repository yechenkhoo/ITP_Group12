const compareCheckboxes = document.querySelectorAll('.compare-checkbox');
const compareBtn = document.getElementById('compareBtn');

const deleteForm = document.getElementById('deleteForm');
const deleteBtn = document.getElementById('deleteBtn');
const deleteInput = document.getElementById('deleteVideoIds');

const selectedCount = document.getElementById('selectedCount');

// Key for storing selected video IDs in sessionStorage
const SELECTED_VIDEO_IDS_KEY = 'selectedVideoIds';

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
    // openTab uses replaceState, which does NOT cause a full page reload.
    // Therefore, sessionStorage persistence for selections will naturally work across tabs.
    window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`);
}

// Initialize the correct tab on page load and manage persistence
document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const initialTab = params.get('tab') || 'tab1'; // Default to tab1 if no tab parameter
    openTab(null, initialTab); // Pass null for event as it's not a user click

    // --- Persistence Reset Logic ---
    // Get the base URL of the current page (e.g., "http://example.com/data-space/videos")
    const currentPageBaseUrl = window.location.origin + window.location.pathname.split('?')[0];
    // Get the referrer (the URL of the page that linked to the current page)
    const referrerUrl = document.referrer;

    // Check if the referrer URL does NOT start with the current page's base URL.
    // This condition means we're coming from an "external" page (not within the data space itself).
    if (referrerUrl && !referrerUrl.startsWith(currentPageBaseUrl)) {
        sessionStorage.removeItem(SELECTED_VIDEO_IDS_KEY); // Clear selections
    }
    // --- End Persistence Reset Logic ---

    // On page load, restore selected checkboxes and update button states
    restoreSelectedCheckboxes();
    updateCompareDeleteButtonState();
});

// Toggle List/Grid
document.getElementById('toggleViewBtn').addEventListener('click', () => {
    const params = new URLSearchParams(window.location.search);
    const currentView = params.get('view') || 'list';
    const currentTab = params.get('tab') || 'tab1'; // Get the current active tab

    params.set('view', currentView === 'list' ? 'grid' : 'list');
    params.set('tab', currentTab); // Preserve the current active tab

    // This line causes a full page reload, which will trigger DOMContentLoaded again.
    // The referrer check in DOMContentLoaded will handle persistence.
    window.location.search = params.toString();
});

// Function to get selected video IDs from sessionStorage
function getStoredSelectedVideoIds() {
    const storedIds = sessionStorage.getItem(SELECTED_VIDEO_IDS_KEY);
    return storedIds ? new Set(JSON.parse(storedIds)) : new Set();
}

// Function to save selected video IDs to sessionStorage
function saveSelectedVideoIds(idsSet) {
    sessionStorage.setItem(SELECTED_VIDEO_IDS_KEY, JSON.stringify(Array.from(idsSet)));
}

// Function to restore checkbox states from sessionStorage on page load
function restoreSelectedCheckboxes() {
    const storedSelectedIds = getStoredSelectedVideoIds();
    compareCheckboxes.forEach(cb => {
        if (storedSelectedIds.has(cb.dataset.videoId)) {
            cb.checked = true;
        } else {
            // Ensure any previously selected checkboxes not in the current stored set are unchecked
            cb.checked = false;
        }
    });
}

function updateCompareDeleteButtonState() {
    // Get currently checked checkboxes on the page
    const currentCheckedBoxes = Array.from(compareCheckboxes).filter(cb => cb.checked);
    const currentCheckedIds = new Set(currentCheckedBoxes.map(cb => cb.dataset.videoId));

    // Get previously selected IDs from sessionStorage (which might have been cleared or retained by DOMContentLoaded)
    const storedSelectedIds = getStoredSelectedVideoIds();

    // Combine current page's checked IDs with stored IDs to get the complete set of selected items
    const allSelectedIds = new Set([...currentCheckedIds, ...storedSelectedIds]);

    // Remove IDs that are *unchecked* on the current page.
    currentCheckedBoxes.forEach(cb => {
        if (!cb.checked) {
            allSelectedIds.delete(cb.dataset.videoId);
        }
    });

    // Update sessionStorage with the combined and cleaned set of all selected IDs
    saveSelectedVideoIds(allSelectedIds);

    const checkedCount = allSelectedIds.size;
    const checkedIdsArray = Array.from(allSelectedIds); // Convert Set to Array for deleteInput value

    // --- compare button logic ---
    if (checkedCount >= 2) {
        compareBtn.disabled = false;
        compareBtn.classList.replace('bg-gray-500', 'bg-[#217346]');
    } else {
        compareBtn.disabled = true;
        compareBtn.classList.replace('bg-[#217346]', 'bg-gray-500');
    }

    // --- delete form logic ---
    if (checkedCount >= 1) {
        deleteForm.classList.remove('hidden');
        deleteBtn.disabled = false;
        deleteInput.value = checkedIdsArray.join(',');
    } else {
        deleteForm.classList.add('hidden');
        deleteBtn.disabled = true;
    }

    // selected-count indicator ---
    if (checkedCount > 0) {
        selectedCount.textContent = `${checkedCount} selected`;
        selectedCount.classList.remove('hidden');
    } else {
        selectedCount.classList.add('hidden');
    }
}

// Wire up each checkbox
compareCheckboxes.forEach(cb => {
    cb.addEventListener('click', e => e.stopPropagation()); // Prevent row click from firing when checkbox is clicked
    cb.addEventListener('change', (event) => {
        const videoId = event.target.dataset.videoId;
        let storedSelectedIds = getStoredSelectedVideoIds(); // Get current stored state

        if (event.target.checked) {
            storedSelectedIds.add(videoId); // Add if checked
        } else {
            storedSelectedIds.delete(videoId); // Remove if unchecked
        }
        saveSelectedVideoIds(storedSelectedIds); // Save updated state to sessionStorage
        updateCompareDeleteButtonState(); // Update UI based on the new total count
    });
});

// Initial state update (important for when you land on a page)
// This is called after DOMContentLoaded has handled the clearing/preserving logic.
updateCompareDeleteButtonState();

document.querySelectorAll('tr[data-href]').forEach(row => {
    row.addEventListener('click', (event) => {
        // Prevent row click if a checkbox within the row was clicked
        if (!event.target.closest('.compare-checkbox')) {
            window.location.href = row.dataset.href;
        }
    });
});