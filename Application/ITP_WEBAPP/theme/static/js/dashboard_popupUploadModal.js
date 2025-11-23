/* ===== Upload modal script with robust diffTimingAlert message handling ===== */

const Uploadmodal = document.getElementById("Uploadmodal");
const closeUploadModalBtn = document.getElementById("closeModalBtn");
const openUploadModalBtn = document.getElementById("openModalBtn"); // may be null if not present

// Detect dataSpace vs coach path
const isStudentDataSpace = window.location.pathname.includes('/home/dataSpace/');

// Build selectors depending on path
const selectors = isStudentDataSpace ? {
    face: {
        input: document.getElementById("videoDBFile_face"),
        dropZone: document.getElementById("dropZoneFaceDB"),
        displayText: document.getElementById("fileDBNameTextFace"),
        icon: document.getElementById("fileDBIconFace"),
        name: "videoDBFile_face"
    },
    dtl: {
        input: document.getElementById("videoDBFile_dtl"),
        dropZone: document.getElementById("dropZoneDTLDB"),
        displayText: document.getElementById("fileDBNameTextDTL"),
        icon: document.getElementById("fileDBIconDTL"),
        name: "videoDBFile_dtl"
    },
    radios: {
        face: document.getElementById("videoTypeFaceDB"),
        dtl: document.getElementById("videoTypeDTLDB"),
        both: document.getElementById("videoTypeBothDB")
    },
    syncReminder: document.getElementById("syncReminderDB"),
    diffTimingAlert: document.getElementById("diffTimingAlertDB") // NEW SELECTOR
} : {
    face: {
        input: document.getElementById("videoFile_face"),
        dropZone: document.getElementById("dropZoneFaceCoach"),
        displayText: document.getElementById("fileNameTextFace"),
        icon: document.getElementById("fileIconFace"),
        name: "videoFile_face"
    },
    dtl: {
        input: document.getElementById("videoFile_dtl"),
        dropZone: document.getElementById("dropZoneDTLCoach"),
        displayText: document.getElementById("fileNameTextDTL"),
        icon: document.getElementById("fileIconDTL"),
        name: "videoFile_dtl"
    },
    radios: {
        face: document.getElementById("videoTypeFaceCoach"),
        dtl: document.getElementById("videoTypeDTLCoach"),
        both: document.getElementById("videoTypeBothCoach")
    },
    syncReminder: document.getElementById("syncReminderCoach"),
    diffTimingAlert: document.getElementById("diffTimingAlertCoach") // NEW SELECTOR
};

// Global variables to store video durations
let faceDuration = null;
let dtlDuration = null;

// helper
function elExists(e){ return e !== null && e !== undefined; }

/* --- NEW: cache the alert message element and its default HTML --- */
if (elExists(selectors.diffTimingAlert)) {
    selectors.diffTimingAlertMsg = selectors.diffTimingAlert.querySelector('.diffTimingAlertMsg');
    selectors.diffTimingAlertMsgDefault = selectors.diffTimingAlertMsg ? selectors.diffTimingAlertMsg.innerHTML : '';
}

/**
 * Asynchronously loads a video file and returns its duration in seconds.
 * @param {File} file - The video file.
 * @returns {Promise<number>} - A promise that resolves with the video duration (number), or null on error.
 */
function getVideoDuration(file) {
    return new Promise((resolve) => {
        if (!file || !file.type.startsWith('video/')) {
            return resolve(null);
        }

        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = function() {
            window.URL.revokeObjectURL(video.src);
            resolve(video.duration);
        };
        video.onerror = function() {
            console.error("Error loading video metadata for duration check.");
            resolve(null);
        };
        video.src = URL.createObjectURL(file);
    });
}

/**
 * Shows an alert element and automatically hides it after a specified duration.
 * @param {HTMLElement} alertElement - The element to show and hide.
 * @param {number} [durationInMs=3000] - The time in milliseconds to display the alert.
 */
function showTimedAlert(alertElement, durationInMs = 3000) {
    if (elExists(alertElement)) {
        // 1. Show the alert
        alertElement.classList.remove('hidden');

        // 2. Set a timeout to hide it
        setTimeout(() => {
            alertElement.classList.add('hidden');
        }, durationInMs);
    }
}


// Initialize: default state - Face On visible, DTL hidden
function initializeState(){
    // Face visible
    if (selectors.face && elExists(selectors.face.dropZone)) selectors.face.dropZone.classList.remove('hidden');
    if (selectors.face && elExists(selectors.face.input)) selectors.face.input.setAttribute('name', selectors.face.name);

    // DTL hidden & keep name (we'll remove name if not active)
    if (selectors.dtl && elExists(selectors.dtl.dropZone)) selectors.dtl.dropZone.classList.add('hidden');
    if (selectors.dtl && elExists(selectors.dtl.input)) {
        // remove name so it doesn't submit by default
        selectors.dtl.input.removeAttribute('name');
    }
    
    // HIDE SYNC REMINDER by default
    if (elExists(selectors.syncReminder)) selectors.syncReminder.classList.add('hidden');
    
    // HIDE NEW ALERT by default and ensure its default message is used
    if (elExists(selectors.diffTimingAlert)) {
        selectors.diffTimingAlert.classList.add('hidden');
        if (selectors.diffTimingAlertMsg && selectors.diffTimingAlertMsgDefault !== undefined) {
            selectors.diffTimingAlertMsg.innerHTML = selectors.diffTimingAlertMsgDefault;
        }
    }

    // radios default to face selected
    if (selectors.radios && selectors.radios.face) selectors.radios.face.checked = true;
    if (selectors.radios && selectors.radios.dtl) selectors.radios.dtl.checked = false;
    if (selectors.radios && selectors.radios.both) selectors.radios.both.checked = false;
}
initializeState();

// modal open/close
if (openUploadModalBtn) {
    openUploadModalBtn.addEventListener('click', () => {
        Uploadmodal.classList.remove('hidden');
    });
}
window.addEventListener('click', (e) => {
    if (e.target === Uploadmodal) {
        resetFileInput();
        Uploadmodal.classList.add('hidden');
    }
});
if (closeUploadModalBtn) {
    closeUploadModalBtn.addEventListener('click', () => {
        resetFileInput();
        Uploadmodal.classList.add('hidden');
    });
}

// reset both inputs/displays and restore default single Face On submission name
function resetFileInput(){
    try {
        if (selectors.face && elExists(selectors.face.input)) {
            selectors.face.input.value = '';
            if (elExists(selectors.face.displayText)) selectors.face.displayText.textContent = 'No file selected';
            if (elExists(selectors.face.icon)) selectors.face.icon.classList.add('hidden');
            selectors.face.input.setAttribute('name', selectors.face.name);
        }
        if (selectors.dtl && elExists(selectors.dtl.input)) {
            selectors.dtl.input.value = '';
            if (elExists(selectors.dtl.displayText)) selectors.dtl.displayText.textContent = 'No file selected';
            if (elExists(selectors.dtl.icon)) selectors.dtl.icon.classList.add('hidden');
            selectors.dtl.input.removeAttribute('name');
        }
        
        // Reset global duration variables
        faceDuration = null;
        dtlDuration = null;

        // show face, hide dtl, reset radios
        if (selectors.face && elExists(selectors.face.dropZone)) selectors.face.dropZone.classList.remove('hidden');
        if (selectors.dtl && elExists(selectors.dtl.dropZone)) selectors.dtl.dropZone.classList.add('hidden');
        
        // HIDE SYNC REMINDER on reset
        if (elExists(selectors.syncReminder)) selectors.syncReminder.classList.add('hidden');
        
        // HIDE NEW ALERT on reset AND restore default message
        if (elExists(selectors.diffTimingAlert)) {
            selectors.diffTimingAlert.classList.add('hidden');

            if (selectors.diffTimingAlertMsg && selectors.diffTimingAlertMsgDefault !== undefined) {
                selectors.diffTimingAlertMsg.innerHTML = selectors.diffTimingAlertMsgDefault;
            }
        }

        if (selectors.radios && selectors.radios.face) selectors.radios.face.checked = true;
        if (selectors.radios && selectors.radios.dtl) selectors.radios.dtl.checked = false;
        if (selectors.radios && selectors.radios.both) selectors.radios.both.checked = false;
    } catch(err) {
        console.warn("resetFileInput error:", err);
    }
}

// file change handlers - MODIFIED to asynchronously get duration
if (selectors.face && elExists(selectors.face.input)) {
    selectors.face.input.addEventListener('change', async () => {
        const file = selectors.face.input.files[0];
        if (file) {
            if (elExists(selectors.face.displayText)) selectors.face.displayText.textContent = file.name;
            if (elExists(selectors.face.icon)) selectors.face.icon.classList.remove('hidden');
            faceDuration = await getVideoDuration(file); // Store duration
            
            // Hide alert on file change and restore default message
            if (elExists(selectors.diffTimingAlert)) {
                selectors.diffTimingAlert.classList.add('hidden');
                if (selectors.diffTimingAlertMsg && selectors.diffTimingAlertMsgDefault !== undefined) {
                    selectors.diffTimingAlertMsg.innerHTML = selectors.diffTimingAlertMsgDefault;
                }
            }
        } else {
            if (elExists(selectors.face.displayText)) selectors.face.displayText.textContent = 'No file selected';
            if (elExists(selectors.face.icon)) selectors.face.icon.classList.add('hidden');
            faceDuration = null;
        }
    });
}
if (selectors.dtl && elExists(selectors.dtl.input)) {
    selectors.dtl.input.addEventListener('change', async () => {
        const file = selectors.dtl.input.files[0];
        if (file) {
            if (elExists(selectors.dtl.displayText)) selectors.dtl.displayText.textContent = file.name;
            if (elExists(selectors.dtl.icon)) selectors.dtl.icon.classList.remove('hidden');
            dtlDuration = await getVideoDuration(file); // Store duration
            
            // Hide alert on file change and restore default message
            if (elExists(selectors.diffTimingAlert)) {
                selectors.diffTimingAlert.classList.add('hidden');
                if (selectors.diffTimingAlertMsg && selectors.diffTimingAlertMsgDefault !== undefined) {
                    selectors.diffTimingAlertMsg.innerHTML = selectors.diffTimingAlertMsgDefault;
                }
            }
        } else {
            if (elExists(selectors.dtl.displayText)) selectors.dtl.displayText.textContent = 'No file selected';
            if (elExists(selectors.dtl.icon)) selectors.dtl.icon.classList.add('hidden');
            dtlDuration = null;
        }
    });
}

// drag & drop helper - MODIFIED to trigger 'change' event
function addDragDropHandlers(zoneEl, inputEl, displayTextEl, iconEl){
    if(!zoneEl) return;
    zoneEl.addEventListener('dragover', (e) => {
        e.preventDefault();
        zoneEl.classList.add('border-indigo-500', 'bg-indigo-50');
    });
    zoneEl.addEventListener('dragleave', () => {
        zoneEl.classList.remove('border-indigo-500', 'bg-indigo-50');
    });
    zoneEl.addEventListener('drop', (e) => {
        e.preventDefault();
        zoneEl.classList.remove('border-indigo-500', 'bg-indigo-50');
        const files = e.dataTransfer.files;
        if (files.length > 0 && inputEl) {
            // Assign files and trigger the 'change' event to run the duration check
            inputEl.files = files;
            inputEl.dispatchEvent(new Event('change'));
        } else {
            if (iconEl) iconEl.classList.add('hidden');
            // If dropping no file, manually call change event to reset
            inputEl.dispatchEvent(new Event('change'));
        }
    });
}
if (selectors.face && elExists(selectors.face.dropZone)) addDragDropHandlers(selectors.face.dropZone, selectors.face.input, selectors.face.displayText, selectors.face.icon);
if (selectors.dtl && elExists(selectors.dtl.dropZone)) addDragDropHandlers(selectors.dtl.dropZone, selectors.dtl.input, selectors.dtl.displayText, selectors.dtl.icon);

// Radio change handling: face, dtl, both
function handleRadioChange(){
    const faceChecked = selectors.radios && selectors.radios.face && selectors.radios.face.checked;
    const dtlChecked = selectors.radios && selectors.radios.dtl && selectors.radios.dtl.checked;
    const bothChecked = selectors.radios && selectors.radios.both && selectors.radios.both.checked;

    // Ensure the alert is hidden when the radio choice changes away from 'both'
    if (elExists(selectors.diffTimingAlert)) {
        selectors.diffTimingAlert.classList.add('hidden');
        // restore default message
        if (selectors.diffTimingAlertMsg && selectors.diffTimingAlertMsgDefault !== undefined) {
            selectors.diffTimingAlertMsg.innerHTML = selectors.diffTimingAlertMsgDefault;
        }
    }

    if (bothChecked) {
        // show both drop zones
        if (selectors.face && elExists(selectors.face.dropZone)) selectors.face.dropZone.classList.remove('hidden');
        if (selectors.dtl && elExists(selectors.dtl.dropZone)) selectors.dtl.dropZone.classList.remove('hidden');
        
        // SHOW SYNC REMINDER
        if (elExists(selectors.syncReminder)) selectors.syncReminder.classList.remove('hidden');

        // ✅ FIX: Set input names to 'face_on_file' and 'dtl_file' as expected by the dual upload server endpoint (rpi_dual_video_upload in views.py)
        if (selectors.face && elExists(selectors.face.input)) selectors.face.input.setAttribute('name', 'face_on_file');
        if (selectors.dtl && elExists(selectors.dtl.input)) selectors.dtl.input.setAttribute('name', 'dtl_file');
    } else if (faceChecked) {
        // show face only
        if (selectors.face && elExists(selectors.face.dropZone)) selectors.face.dropZone.classList.remove('hidden');
        if (selectors.dtl && elExists(selectors.dtl.dropZone)) selectors.dtl.dropZone.classList.add('hidden');
        
        // HIDE SYNC REMINDER
        if (elExists(selectors.syncReminder)) selectors.syncReminder.classList.add('hidden');

        if (selectors.face && elExists(selectors.face.input)) selectors.face.input.setAttribute('name', selectors.face.name);
        if (selectors.dtl && elExists(selectors.dtl.input)) selectors.dtl.input.removeAttribute('name');
    } else if (dtlChecked) {
        // show dtl only
        if (selectors.face && elExists(selectors.face.dropZone)) selectors.face.dropZone.classList.add('hidden');
        if (selectors.dtl && elExists(selectors.dtl.dropZone)) selectors.dtl.dropZone.classList.remove('hidden');
        
        // HIDE SYNC REMINDER
        if (elExists(selectors.syncReminder)) selectors.syncReminder.classList.add('hidden');

        if (selectors.dtl && elExists(selectors.dtl.input)) selectors.dtl.input.setAttribute('name', selectors.dtl.name);
        if (selectors.face && elExists(selectors.face.input)) selectors.face.input.removeAttribute('name');
    } else {
        // fallback: default to face-only
        if (selectors.face && elExists(selectors.face.dropZone)) selectors.face.dropZone.classList.remove('hidden');
        if (selectors.dtl && elExists(selectors.dtl.dropZone)) selectors.dtl.dropZone.classList.add('hidden');
        
        // HIDE SYNC REMINDER
        if (elExists(selectors.syncReminder)) selectors.syncReminder.classList.add('hidden');

        if (selectors.face && elExists(selectors.face.input)) selectors.face.input.setAttribute('name', selectors.face.name);
        if (selectors.dtl && elExists(selectors.dtl.input)) selectors.dtl.input.removeAttribute('name');
    }
}

// Attach radio listeners
if (selectors.radios) {
    if (selectors.radios.face) selectors.radios.face.addEventListener('change', handleRadioChange);
    if (selectors.radios.dtl) selectors.radios.dtl.addEventListener('change', handleRadioChange);
    if (selectors.radios.both) selectors.radios.both.addEventListener('change', handleRadioChange);
}

// Add form submission listener for validation
const formElement = Uploadmodal.querySelector('form');

if (formElement) {
    formElement.addEventListener('submit', function(e) {
        const bothChecked = selectors.radios && selectors.radios.both && selectors.radios.both.checked;
        const diffTimingAlert = selectors.diffTimingAlert;
        
        // Ensure initial state for the alert is hidden for non-blocking submissions
        if (elExists(diffTimingAlert)) {
             diffTimingAlert.classList.add('hidden');
             // restore default message
             if (selectors.diffTimingAlertMsg && selectors.diffTimingAlertMsgDefault !== undefined) {
                 selectors.diffTimingAlertMsg.innerHTML = selectors.diffTimingAlertMsgDefault;
             }
        }
        
        // 1. Check if "both" is selected
        if (bothChecked) {
            // Check if both files are present
            if (!selectors.face.input.files[0] || !selectors.dtl.input.files[0]) {
                // Let browser handle required attr if present. If not present, stop here (no durations to check).
                return; 
            }

            // 2. Check for duration difference
            const durationTolerance = 0.1; // Allow up to 100ms difference due to floating point inaccuracies
            
            // Handle case where duration metadata couldn't be loaded (e.g., non-video file)
            if (faceDuration === null || dtlDuration === null) {
                if (elExists(selectors.diffTimingAlert)) {
                    const msg = "Could not read duration metadata for one or both videos. Please try re-selecting the files.";
                    if (selectors.diffTimingAlertMsg) selectors.diffTimingAlertMsg.innerHTML = msg;
                    // SHOW ALERT AND SET TIMEOUT
                    showTimedAlert(selectors.diffTimingAlert, 3000);
                }
                e.preventDefault(); // DENY UPLOAD
                return;
            }

            const durationDiff = Math.abs(faceDuration - dtlDuration);

            if (durationDiff > durationTolerance) {
                // Durations are different, deny upload and show alert
                if (elExists(selectors.diffTimingAlert)) {
                    // restore (or use) the default HTML message we cached (preserves <strong>)
                    if (selectors.diffTimingAlertMsg && selectors.diffTimingAlertMsgDefault !== undefined) {
                        selectors.diffTimingAlertMsg.innerHTML = selectors.diffTimingAlertMsgDefault || "Both videos must have the same duration (timing) to proceed.";
                    } else if (selectors.diffTimingAlertMsg) {
                        selectors.diffTimingAlertMsg.innerHTML = "Both videos must have the same duration (timing) to proceed.";
                    }
                    // SHOW ALERT AND SET TIMEOUT
                    showTimedAlert(selectors.diffTimingAlert, 3000);
                }
                e.preventDefault(); // DENY UPLOAD
                return;
            }
        }
        
        // If validation passes (or 'both' wasn't selected), allow submission
    });
}

// initialize the correct view on load
handleRadioChange();