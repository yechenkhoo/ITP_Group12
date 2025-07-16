const compareCheckboxes = document.querySelectorAll('.compare-checkbox');
const compareBtn = document.getElementById('compareBtn');
const compareForm       = document.getElementById('compareForm');
const compareIdsInput   = document.getElementById('compareVideoIds');

const deleteForm = document.getElementById('deleteForm');
const deleteBtn = document.getElementById('deleteBtn');
const deleteInput = document.getElementById('deleteVideoIds');

const selectedCount = document.getElementById('selectedCount');

// Polling specific variables
const POLL_INTERVAL_MS = 5000; // Poll every 5 seconds
const pollingIntervals = {}; // Stores interval IDs for each video being polled

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
    if (checkedCount == 2) {
        compareBtn.disabled = false;
        compareBtn.classList.replace('bg-gray-500', 'bg-[#217346]');
    } else {
        compareBtn.disabled = true;
        compareBtn.classList.replace('bg-[#217346]', 'bg-gray-500');
    }

    compareIdsInput.value = checkedIdsArray.join(',');

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

/**
 * Updates the UI of a specific video element when its status changes.
 * @param {HTMLElement} videoElement The HTML element (tr or div/a) representing the video.
 * @param {object} videoData An object containing updated video data (e.g., status, rawVideoLink).
 */
function updateVideoUI(videoElement, videoData) {
    console.log(`Updating UI for video ${videoData.id} to status: ${videoData.status}`);

    // Update data-video-status attribute
    videoElement.dataset.videoStatus = videoData.status;

    // Get relevant child elements for update
    const statusSpan = videoElement.querySelector('.video-status-text');
    const thumbnailCell = videoElement.querySelector('.video-thumbnail-cell');
    const videoTitle = videoElement.querySelector('.video-title').textContent; // Get the title for alt text

    if (videoData.status === 'Completed') {
        // Update status text and color
        if (statusSpan) {
            statusSpan.textContent = videoData.status;
            statusSpan.classList.remove('bg-red-100', 'text-red-800');
            statusSpan.classList.add('bg-green-100', 'text-green-800');
        }

        // Add checkbox if it's a list item (table row) and not already present
        // Or if it's a grid item that became clickable
        const isGridItem = videoElement.classList.contains('grid-view') || videoElement.tagName === 'A' || videoElement.tagName === 'DIV';

        let checkbox = videoElement.querySelector('.compare-checkbox');
        if (!checkbox) { // Only add if it doesn't exist
            checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.classList.add('compare-checkbox');
            checkbox.dataset.videoId = videoData.id;

            if (isGridItem) {
                checkbox.classList.add('absolute', 'top-2', 'right-2', 'z-10');
                videoElement.prepend(checkbox); // Add as first child for grid items
            } else { // List view (table row)
                const firstCell = videoElement.querySelector('td:first-child');
                if (firstCell) {
                    firstCell.innerHTML = ''; // Clear existing content (which might be empty or 'Processing...')
                    firstCell.appendChild(checkbox);
                }
            }
            // Re-attach event listener for the newly added checkbox
            checkbox.addEventListener('click', e => e.stopPropagation());
            checkbox.addEventListener('change', updateCompareDeleteButtonState);

             // If this video was previously selected (and persisted) before becoming 'Completed', check it
             const storedSelectedIds = getStoredSelectedVideoIds();
             if (storedSelectedIds.has(videoData.id)) {
                 checkbox.checked = true;
             }
        }


        // Update thumbnail
        if (thumbnailCell && videoData.rawVideoLink) {
            const videoThumbnail = document.createElement('video');
            videoThumbnail.classList.add('w-full', 'h-24', 'object-cover', 'rounded', 'mb-2'); // Tailwind classes
            if (!isGridItem) { // Different sizes for list vs grid view
                videoThumbnail.classList.remove('w-full', 'h-24');
                videoThumbnail.classList.add('h-32', 'w-48');
            }

            videoThumbnail.preload = 'metadata';
            videoThumbnail.muted = true;
            videoThumbnail.loop = true;
            videoThumbnail.onmouseover = function() { this.play(); };
            videoThumbnail.onmouseout = function() { this.pause(); this.currentTime = 0; };

            const source = document.createElement('source');
            source.src = videoData.rawVideoLink;
            source.type = 'video/mp4';
            videoThumbnail.appendChild(source);

            // Replace the 'Processing...' div with the new video element
            thumbnailCell.innerHTML = '';
            thumbnailCell.appendChild(videoThumbnail);
        }

        // Make the video element clickable
        const currentStudentID = window.__studentID;
        if (videoElement.tagName === 'DIV' && videoElement.dataset.videoStatus === 'Completed') {
            // If it was a <div> and now completed, change it to an <a>
            const newAnchor = document.createElement('a');
            newAnchor.href = `/home/dataSpace/${currentStudentID}/results/${videoData.id}/`;
            newAnchor.classList.add('relative', 'block', 'bg-white', 'border', 'rounded-lg', 'shadow', 'p-3', 'flex', 'flex-col', 'hover:shadow-md', 'hover:bg-gray-50', 'cursor-pointer', 'max-w-[430px]');
            newAnchor.dataset.videoId = videoData.id;
            newAnchor.dataset.videoStatus = videoData.status;

            // Move all children from the old div to the new anchor
            while (videoElement.firstChild) {
                newAnchor.appendChild(videoElement.firstChild);
            }
            videoElement.replaceWith(newAnchor); // Replace the old div with the new anchor
            videoElement = newAnchor; // Update reference
        } else if (videoElement.tagName === 'TR' && !videoElement.dataset.href) {
            // For list view, add the data-href and clickable classes
            videoElement.dataset.href = `/home/dataSpace/${currentStudentID}/results/${videoData.id}/`;
            videoElement.classList.add('hover:bg-gray-50', 'cursor-pointer');
            // Re-attach click listener for the row
            videoElement.addEventListener('click', (event) => {
                if (!event.target.closest('.compare-checkbox')) {
                    window.location.href = videoElement.dataset.href;
                }
            });
        }
    } else {
        // Handle other statuses if necessary (e.g., Error)
        console.warn(`Video ${videoData.id} status is ${videoData.status}, no special UI update handled for non-completed.`);
    }

    // After updating UI for a specific video, re-evaluate button states
    updateCompareDeleteButtonState();
}

// Function to start polling for all 'Processing' videos
function startStatusPolling() {
    // Select all video elements that might need status updates
    // Use data-video-id and data-video-status attributes
    document.querySelectorAll('[data-video-id][data-video-status="Processing"]').forEach(videoElement => {
        const videoId = videoElement.dataset.videoId;

        // If already polling for this video, skip
        if (pollingIntervals[videoId]) {
            return;
        }

        console.log(`Starting polling for video: ${videoId}`);

        // Set up polling interval
        const intervalId = setInterval(async () => {
            try {
                // Use window.__studentID for the student ID from Django template
                const studentID = window.__studentID; // Ensure studentID is available
                if (!studentID) {
                    console.error("studentID not found. Cannot poll for video status.");
                    clearInterval(pollingIntervals[videoId]);
                    delete pollingIntervals[videoId];
                    return;
                }
                const response = await fetch(`/home/dataSpace/${studentID}/check_video_status_ajax/${videoId}/`);
                const data = await response.json();

                if (data.status === 'success' && data.video_status === 'Completed') {
                    console.log(`Video ${videoId} is now Completed!`);
                    clearInterval(pollingIntervals[videoId]); // Stop polling
                    delete pollingIntervals[videoId]; // Remove from tracking

                    // Update UI for this video
                    updateVideoUI(videoElement, {
                        id: videoId,
                        status: data.video_status,
                        rawVideoLink: data.rawVideoLink, // Ensure this is passed from backend
                    });

                    // Optional: If you want to automatically move completed videos from 'Processing' tab
                    // to 'Completed' tab without a full reload, this would be complex.
                    // A full page reload (e.g., window.location.reload()) would achieve this,
                    // but it would interrupt user experience.
                    // For now, the UI within the 'Processing' tab will update.
                    // If the user navigates to the 'All' or 'Completed' tab, they will see it.

                } else if (data.status === 'error') {
                    console.error(`Error checking status for video ${videoId}: ${data.message}`);
                    clearInterval(pollingIntervals[videoId]);
                    delete pollingIntervals[videoId];
                }
                // If still processing, do nothing and wait for next poll
            } catch (error) {
                console.error(`Fetch error for video ${videoId}:`, error);
                clearInterval(pollingIntervals[videoId]);
                delete pollingIntervals[videoId];
            }
        }, POLL_INTERVAL_MS);

        pollingIntervals[videoId] = intervalId; // Store the interval ID
    });
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
startStatusPolling()

document.querySelectorAll('tr[data-href]').forEach(row => {
    row.addEventListener('click', (event) => {
        // Prevent row click if a checkbox within the row was clicked
        if (!event.target.closest('.compare-checkbox')) {
            window.location.href = row.dataset.href;
        }
    });
});