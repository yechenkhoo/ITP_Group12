const compareCheckboxes = document.querySelectorAll('.compare-checkbox');
const compareBtn = document.getElementById('compareBtn');
const compareForm = document.getElementById('compareForm');
const compareIdsInput = document.getElementById('compareVideoIds');

const deleteForm = document.getElementById('deleteForm');
const deleteBtn = document.getElementById('deleteBtn');
const deleteInput = document.getElementById('deleteVideoIds');

const selectedCount = document.getElementById('selectedCount');
const selectAllCheckboxes = document.querySelectorAll('.compare-checkbox-all'); // Select all "select all" checkboxes

// --- [NEW] Modal Elements ---
const deleteConfirmModal = document.getElementById('delete-confirm-modal'); //
const cancelDeleteBtn = document.getElementById('cancel-delete-btn'); //
const confirmDeleteBtn = document.getElementById('confirm-delete-btn'); //
const deleteModalMessage = document.getElementById('delete-modal-message'); //


// Polling specific variables
const POLL_INTERVAL_MS = 5000; // Poll every 5 seconds
const pollingIntervals = {}; // Stores interval IDs for each video being polled

// Key for storing selected video IDs in sessionStorage
const SELECTED_VIDEO_IDS_KEY = 'selectedVideoIds'; //

// --- [NEW] Modal Control Functions ---
function showDeleteModal() { //
    if (!deleteConfirmModal) return; //
    deleteConfirmModal.classList.remove('hidden'); //
    setTimeout(() => { // Allow display property to apply before starting transition
        deleteConfirmModal.classList.remove('opacity-0'); //
        // Access the inner div for the scale transform
        deleteConfirmModal.querySelector('div').classList.remove('scale-95'); //
    }, 10); //
}

function hideDeleteModal() { //
    if (!deleteConfirmModal) return; //
    deleteConfirmModal.classList.add('opacity-0'); //
    deleteConfirmModal.querySelector('div').classList.add('scale-95'); //
    setTimeout(() => { // Wait for transition to finish before hiding
        deleteConfirmModal.classList.add('hidden'); //
    }, 300); // Duration should match the CSS transition duration
}

// Function to open a specific tab and update URL
function openTab(event, tabId) { //
    var i, tabcontent, tablinks; //

    // Hide all tab contents
    tabcontent = document.getElementsByClassName("tab-content"); //
    for (i = 0; i < tabcontent.length; i++) { //
        tabcontent[i].classList.add("hidden"); //
    }

    // Remove active class from all tab buttons
    tablinks = document.querySelectorAll("[data-tab-target]"); //
    for (i = 0; i < tablinks.length; i++) { //
        tablinks[i].classList.remove("border-blue-500", "text-blue-500"); //
        tablinks[i].classList.add("border-transparent", "hover:text-gray-600", "hover:border-gray-300"); //
    }

    // Show the current tab content
    document.getElementById(tabId).classList.remove("hidden"); //

    // Add active class to the current tab button
    if (event && event.currentTarget) { // Check if event and currentTarget exist (for user clicks)
        event.currentTarget.classList.add("border-blue-500", "text-blue-500"); //
        event.currentTarget.classList.remove("border-transparent", "hover:text-gray-600", "hover:border-gray-300"); //
    } else { // For programmatic calls (like initial load)
        const correspondingTabButton = document.querySelector(`a[href*="tab=${tabId}"]`); // Modified selector
        if (correspondingTabButton) { //
            correspondingTabButton.classList.add("border-blue-500", "text-blue-500"); //
            correspondingTabButton.classList.remove("border-transparent", "hover:text-gray-600", "hover:border-gray-300"); //
        }
    }

    // Update URL to include the active tab
    const params = new URLSearchParams(window.location.search); //
    params.set('tab', tabId); //
    // Preserve other parameters like 'view' and 'sort'
    const currentView = params.get('view') || 'list'; //
    const currentSort = params.get('sort') || ''; //
    params.set('view', currentView); //
    if (currentSort) { //
        params.set('sort', currentSort); //
    }
    // openTab uses replaceState, which does NOT cause a full page reload.
    // Therefore, sessionStorage persistence for selections will naturally work across tabs.
    window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`); //
}

// Initialize the correct tab on page load and manage persistence
document.addEventListener('DOMContentLoaded', () => { //
    const params = new URLSearchParams(window.location.search); //
    const initialTab = params.get('tab') || 'tab1'; // Default to tab1 if no tab parameter
    openTab(null, initialTab); // Pass null for event as it's not a user click

    // --- Persistence Reset Logic ---
    // Get the base URL of the current page (e.g., "http://example.com/data-space/videos")
    const currentPageBaseUrl = window.location.origin + window.location.pathname.split('?')[0]; //
    // Get the referrer (the URL of the page that linked to the current page)
    const referrerUrl = document.referrer; //

    // Check if the referrer URL does NOT start with the current page's base URL.
    // This condition means we're coming from an "external" page (not within the data space itself).
    if (referrerUrl && !referrerUrl.startsWith(currentPageBaseUrl)) { //
        sessionStorage.removeItem(SELECTED_VIDEO_IDS_KEY); // Clear selections
    }
    // --- End Persistence Reset Logic ---

    // On page load, restore selected checkboxes and update button states
    restoreSelectedCheckboxes(); //
    updateCompareDeleteButtonState(); //
});

// Toggle List/Grid
document.getElementById('toggleViewBtn').addEventListener('click', () => {
    // Clear selections from session storage before reloading the page for the new view.
    // This prevents the "Delete" button and selection count from incorrectly persisting across view changes.
    sessionStorage.removeItem(SELECTED_VIDEO_IDS_KEY);

    const params = new URLSearchParams(window.location.search);
    const currentView = params.get('view') || 'list';
    const currentTab = params.get('tab') || 'tab1';

    params.set('view', currentView === 'list' ? 'grid' : 'list');
    params.set('tab', currentTab);

    // This line causes a full page reload, which will trigger DOMContentLoaded again.
    // The referrer check in DOMContentLoaded will handle persistence.
    window.location.search = params.toString();
});

// Function to get selected video IDs from sessionStorage
function getStoredSelectedVideoIds() { //
    const storedIds = sessionStorage.getItem(SELECTED_VIDEO_IDS_KEY); //
    return storedIds ? new Set(JSON.parse(storedIds)) : new Set(); //
}

// Function to save selected video IDs to sessionStorage
function saveSelectedVideoIds(idsSet) { //
    sessionStorage.setItem(SELECTED_VIDEO_IDS_KEY, JSON.stringify(Array.from(idsSet))); //
}

// Function to restore checkbox states from sessionStorage on page load
function restoreSelectedCheckboxes() { //
    const storedSelectedIds = getStoredSelectedVideoIds(); //
    compareCheckboxes.forEach(cb => { //
        if (storedSelectedIds.has(cb.dataset.videoId)) { //
            cb.checked = true; //
        } else {
            // Ensure any previously selected checkboxes not in the current stored set are unchecked
            cb.checked = false; //
        }
    });
}

function updateCompareDeleteButtonState() { //
    // Get currently checked checkboxes on the page
    const currentCheckedBoxes = Array.from(compareCheckboxes).filter(cb => cb.checked); //
    const currentCheckedIds = new Set(currentCheckedBoxes.map(cb => cb.dataset.videoId)); //

    // Get previously selected IDs from sessionStorage
    const storedSelectedIds = getStoredSelectedVideoIds(); //

    // Combine to get the complete set of selected items
    const allSelectedIds = new Set([...currentCheckedIds, ...storedSelectedIds]); //

    // Remove any IDs unchecked on the current page
    Array.from(allSelectedIds).forEach(id => { //
        const cb = document.querySelector(`.compare-checkbox[data-video-id="${id}"]`); //
        if (cb && !cb.checked) { //
            allSelectedIds.delete(id); //
        }
    });

    // Persist the updated full set
    saveSelectedVideoIds(allSelectedIds); //

    // --- DELETE logic stays the same (allSelectedIds) ---
    const allSelectedArray = Array.from(allSelectedIds); //
    if (allSelectedArray.length >= 1) { //
        deleteForm.classList.remove('hidden'); //
        deleteBtn.disabled = false; //
        deleteInput.value = allSelectedArray.join(','); //
    } else {
        deleteForm.classList.add('hidden'); //
        deleteBtn.disabled = true; //
    }

    // --- COMPARE logic only uses non-processing and non-failed videos ---
    // Filter for IDs whose element is not in 'Processing' or 'Failed' status
    const compareEligibleIds = allSelectedArray.filter(id => { //
        const el = document.querySelector(`[data-video-id="${id}"]`); //
        return el && el.dataset.videoStatus !== 'Processing' && el.dataset.videoStatus !== 'Failed'; //
    });

    if (compareEligibleIds.length === 2) { //
        compareBtn.disabled = false; //
        compareBtn.classList.replace('bg-gray-500', 'bg-[#217346]'); //
    } else {
        compareBtn.disabled = true; //
        compareBtn.classList.replace('bg-[#217346]', 'bg-gray-500'); //
    }
    // Only pass the eligible IDs into the compare form
    compareIdsInput.value = compareEligibleIds.join(','); //

    // --- SELECTED COUNT indicator (show total selected) ---
    selectedCount.textContent = `${allSelectedArray.length} selected`; //
    if (allSelectedArray.length > 0) { //
        selectedCount.classList.remove('hidden'); //
    } else {
        selectedCount.classList.add('hidden'); //
    }

    // --- [NEW] Sync 'Select All' checkbox state ---
    selectAllCheckboxes.forEach(selectAllCb => { //
        const activeTabContent = document.querySelector('.tab-content:not(.hidden)'); //
        let individualCheckboxes = []; //

        if (activeTabContent) { //
            // Broadly select all .compare-checkboxes within the active tab's content.
            // This handles cases where .list-view or .grid-view might not be the direct parent,
            // or if specific tabs have different structures.
            individualCheckboxes = Array.from(activeTabContent.querySelectorAll('.compare-checkbox')); //
        }

        // Filter out checkboxes that are hidden (e.g., from other pages via pagination)
        // if you only want to consider visible checkboxes for "Select All"
        const visibleIndividualCheckboxes = individualCheckboxes.filter(cb => { //
            // A simple way to check visibility might be to check if its parent row/card is displayed
            return cb.offsetParent !== null; // offsetParent is null for hidden elements
        });

        // Ensure there are visible checkboxes to check and all of them are, in fact, checked
        const allChecked = visibleIndividualCheckboxes.length > 0 && visibleIndividualCheckboxes.every(cb => cb.checked); //
        selectAllCb.checked = allChecked; //
    });
}

/**
 * Updates the UI of a specific video element when its status changes.
 * @param {HTMLElement} videoElement The HTML element (tr or div/a) representing the video.
 * @param {object} videoData An object containing updated video data (e.g., status, rawVideoLink).
 */
function updateVideoUI(videoElement, videoData) { //
    console.log(`Updating UI for video ${videoData.id} to status: ${videoData.status}`); //

    // Update data-video-status attribute
    videoElement.dataset.videoStatus = videoData.status; //

    // Get relevant child elements for update
    const statusSpan = videoElement.querySelector('.video-status-text'); //
    const thumbnailCell = videoElement.querySelector('.video-thumbnail-cell'); //
    const videoTitle = videoElement.querySelector('.video-title').textContent; // Get the title for alt text

    if (videoData.status === 'Completed') { //
        // Update status text and color
        if (statusSpan) { //
            statusSpan.textContent = videoData.status; //
            statusSpan.classList.remove('bg-red-100', 'text-red-800'); //
            statusSpan.classList.add('bg-green-100', 'text-green-800'); //
        }

        // Add checkbox if it's a list item (table row) and not already present
        // Or if it's a grid item that became clickable
        const isGridItem = videoElement.classList.contains('grid-view') || videoElement.tagName === 'A' || videoElement.tagName === 'DIV'; //

        let checkbox = videoElement.querySelector('.compare-checkbox'); //
        if (!checkbox) { // Only add if it doesn't exist
            checkbox = document.createElement('input'); //
            checkbox.type = 'checkbox'; //
            checkbox.classList.add('compare-checkbox'); //
            checkbox.dataset.videoId = videoData.id; //

            if (isGridItem) { //
                checkbox.classList.add('absolute', 'top-2', 'right-2', 'z-10'); //
                videoElement.prepend(checkbox); // Add as first child for grid items
            } else { // List view (table row)
                const firstCell = videoElement.querySelector('td:first-child'); //
                if (firstCell) { //
                    firstCell.innerHTML = ''; // Clear existing content (which might be empty or 'Processing...')
                    firstCell.appendChild(checkbox); //
                }
            }
            // Re-attach event listener for the newly added checkbox
            checkbox.addEventListener('click', e => e.stopPropagation()); //
            checkbox.addEventListener('change', updateCompareDeleteButtonState); //

            // If this video was previously selected (and persisted) before becoming 'Completed', check it
            const storedSelectedIds = getStoredSelectedVideoIds(); //
            if (storedSelectedIds.has(videoData.id)) { //
                checkbox.checked = true; //
            }
        }


        // Update thumbnail
        if (thumbnailCell && videoData.rawVideoLink) { //
            const videoThumbnail = document.createElement('video'); //
            videoThumbnail.classList.add('w-full', 'h-24', 'object-cover', 'rounded', 'mb-2'); // Tailwind classes
            if (!isGridItem) { // Different sizes for list vs grid view
                videoThumbnail.classList.remove('w-full', 'h-24'); //
                videoThumbnail.classList.add('h-32', 'w-48'); //
            }

            videoThumbnail.preload = 'metadata'; //
            videoThumbnail.muted = true; //
            videoThumbnail.loop = true; //
            videoThumbnail.onmouseover = function () { //
                this.play(); //
            }; //
            videoThumbnail.onmouseout = function () { //
                this.pause(); //
                this.currentTime = 0; //
            }; //

            const source = document.createElement('source'); //
            source.src = videoData.rawVideoLink; //
            source.type = 'video/mp4'; //
            videoThumbnail.appendChild(source); //

            // Replace the 'Processing...' div with the new video element
            thumbnailCell.innerHTML = ''; //
            thumbnailCell.appendChild(videoThumbnail); //
        }

        // Make the video element clickable
        const currentStudentID = window.__studentID; //
        if (videoElement.tagName === 'DIV' && videoElement.dataset.videoStatus === 'Completed') { //
            // If it was a <div> and now completed, change it to an <a>
            const newAnchor = document.createElement('a'); //
            newAnchor.href = `/home/dataSpace/${currentStudentID}/results/${videoData.id}/`; //
            newAnchor.classList.add('relative', 'block', 'bg-white', 'border', 'rounded-lg', 'shadow', 'p-3', 'flex', 'flex-col', 'hover:shadow-md', 'hover:bg-gray-50', 'cursor-pointer', 'max-w-[430px]'); //
            newAnchor.dataset.videoId = videoData.id; //
            newAnchor.dataset.videoStatus = videoData.status; //

            // Move all children from the old div to the new anchor
            while (videoElement.firstChild) { //
                newAnchor.appendChild(videoElement.firstChild); //
            }
            videoElement.replaceWith(newAnchor); // Replace the old div with the new anchor
            videoElement = newAnchor; // Update reference
        } else if (videoElement.tagName === 'TR' && !videoElement.dataset.href) { //
            // For list view, add the data-href and clickable classes
            videoElement.dataset.href = `/home/dataSpace/${currentStudentID}/results/${videoData.id}/`; //
            videoElement.classList.add('hover:bg-gray-50', 'cursor-pointer'); //
            // Re-attach click listener for the row
            videoElement.addEventListener('click', (event) => { //
                if (!event.target.closest('.compare-checkbox')) { //
                    window.location.href = videoElement.dataset.href; //
                }
            });
        }
    } else if (videoData.status === 'Failed') { //
        // Update status text and color for Failed
        if (statusSpan) { //
            statusSpan.textContent = videoData.status; //
            statusSpan.classList.remove('bg-green-100', 'text-green-800'); //
            statusSpan.classList.add('bg-red-100', 'text-red-800'); //
        }
        // Update thumbnail to indicate failure
        if (thumbnailCell) { //
            thumbnailCell.innerHTML = `
                <div class="h-28 w-40 md:h-32 md:w-48 object-cover rounded bg-gray-200 flex items-center justify-center text-red-500 text-center">
                    Failed
                </div>
            `; //
        }
        // Ensure no clickability for failed items
        videoElement.removeAttribute('data-href'); //
        videoElement.classList.remove('hover:bg-gray-50', 'cursor-pointer'); //
        videoElement.removeEventListener('click', (event) => { //
            if (!event.target.closest('.compare-checkbox')) { //
                window.location.href = videoElement.dataset.href; //
            }
        });
    } else {
        // Handle other statuses if necessary (e.g., Error, if 'Failed' isn't the final error state)
        console.warn(`Video ${videoData.id} status is ${videoData.status}, no special UI update handled for non-completed/non-failed.`); //
    }

    // After updating UI for a specific video, re-evaluate button states
    updateCompareDeleteButtonState(); //
}

// Function to start polling for all 'Processing' videos
function startStatusPolling() { //
    // Select all video elements that might need status updates
    // Use data-video-id and data-video-status attributes
    document.querySelectorAll('[data-video-id][data-video-status="Processing"]').forEach(videoElement => { //
        const videoId = videoElement.dataset.videoId; //

        // If already polling for this video, skip
        if (pollingIntervals[videoId]) { //
            return; //
        }

        console.log(`Starting polling for video: ${videoId}`); //

        // Set up polling interval
        const intervalId = setInterval(async () => { //
            try {
                // Use window.__studentID for the student ID from Django template
                const studentID = window.__studentID; // Ensure studentID is available
                if (!studentID) { //
                    console.error("studentID not found. Cannot poll for video status."); //
                    clearInterval(pollingIntervals[videoId]); //
                    delete pollingIntervals[videoId]; //
                    return; //
                }
                const response = await fetch(`/home/dataSpace/${studentID}/check_video_status_ajax/${videoId}/`); //
                const data = await response.json(); //

                if (data.status === 'success' && (data.video_status === 'Completed' || data.video_status === 'Failed')) { //
                    console.log(`Video ${videoId} is now ${data.video_status}!`); //
                    clearInterval(pollingIntervals[videoId]); // Stop polling
                    delete pollingIntervals[videoId]; // Remove from tracking

                    // Update UI for this video
                    updateVideoUI(videoElement, { //
                        id: videoId, //
                        status: data.video_status, //
                        rawVideoLink: data.rawVideoLink, // Ensure this is passed from backend
                    });

                } else if (data.status === 'error') { //
                    console.error(`Error checking status for video ${videoId}: ${data.message}`); //
                    clearInterval(pollingIntervals[videoId]); //
                    delete pollingIntervals[videoId]; //
                    // Potentially update UI to 'Failed' if the error means permanent failure
                    updateVideoUI(videoElement, { //
                        id: videoId, //
                        status: 'Failed', // Set status to Failed on error
                        rawVideoLink: null, // No raw video link on failure
                    });
                }
            } catch (error) { //
                console.error(`Fetch error for video ${videoId}:`, error); //
                clearInterval(pollingIntervals[videoId]); //
                delete pollingIntervals[videoId]; //
                // Update UI to 'Failed' if there's a fetch error
                updateVideoUI(videoElement, { //
                    id: videoId, //
                    status: 'Failed', // Set status to Failed on fetch error
                    rawVideoLink: null, // No raw video link on failure
                });
            }
        }, POLL_INTERVAL_MS);

        pollingIntervals[videoId] = intervalId; // Store the interval ID
    });
}

// --- Event listener for 'Select All' checkboxes ---
selectAllCheckboxes.forEach(selectAllCb => { //
    selectAllCb.addEventListener('change', (event) => { //
        const isChecked = event.target.checked; //
        const activeTabContent = document.querySelector('.tab-content:not(.hidden)'); //
        let checkboxesInView = []; //

        if (activeTabContent) { //
            // Use the same broad selection as proposed for Issue 1
            checkboxesInView = Array.from(activeTabContent.querySelectorAll('.compare-checkbox')); //
        }

        let storedSelectedIds = getStoredSelectedVideoIds(); // Get current stored state

        checkboxesInView.forEach(cb => { //
            const videoId = cb.dataset.videoId; //
            if (cb.checked !== isChecked) { //
                cb.checked = isChecked; //
                // Directly add/remove from the storedSelectedIds set
                if (isChecked) { //
                    storedSelectedIds.add(videoId); //
                } else {
                    storedSelectedIds.delete(videoId); //
                }
            }
        });

        saveSelectedVideoIds(storedSelectedIds); // Save updated state once
        updateCompareDeleteButtonState(); // Then update UI
    });
});

// Wire up each checkbox
compareCheckboxes.forEach(cb => { //
    cb.addEventListener('click', e => e.stopPropagation()); // Prevent row click from firing
    cb.addEventListener('change', (event) => { //
        const videoId = event.target.dataset.videoId; //
        let storedSelectedIds = getStoredSelectedVideoIds(); //
        if (event.target.checked) { //
            storedSelectedIds.add(videoId); //
        } else {
            storedSelectedIds.delete(videoId); //
        }
        saveSelectedVideoIds(storedSelectedIds); //
        updateCompareDeleteButtonState(); //
    });
});

// --- [NEW] Intercept Delete Form Submission to show modal ---
deleteForm.addEventListener('submit', (event) => { //
    event.preventDefault(); // Stop the form from submitting immediately

    const videoIds = deleteInput.value.split(','); //
    const numVideos = videoIds.filter(id => id).length; // Filter out empty strings

    // Update modal message based on number of videos
    if (numVideos === 1) { //
        deleteModalMessage.textContent = 'Are you sure you want to delete this video? This action cannot be undone.'; //
    } else {
        deleteModalMessage.textContent = `Are you sure you want to delete these ${numVideos} videos? This action cannot be undone.`; //
    }
    showDeleteModal(); //
});

// --- [NEW] Add listeners for modal buttons ---
if (cancelDeleteBtn) { //
    cancelDeleteBtn.addEventListener('click', () => { //
        hideDeleteModal(); //
    });
}

if (confirmDeleteBtn) { //
    confirmDeleteBtn.addEventListener('click', () => { //
        sessionStorage.removeItem(SELECTED_VIDEO_IDS_KEY); //
        hideDeleteModal(); //
        // Programmatically submit the form. This does not trigger the 'submit' event listener again.
        deleteForm.submit(); //
    });
}


// Initial state update
updateCompareDeleteButtonState(); //
startStatusPolling(); //

document.querySelectorAll('tr[data-href]').forEach(row => { //
    row.addEventListener('click', (event) => { //
        if (!event.target.closest('.compare-checkbox')) { //
            window.location.href = row.dataset.href; //
        }
    });
});