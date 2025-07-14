// dashboard_comments.js (... svg icons for replies aligned with the comments one now! just need to fix back some minor stylings)

// This script forces a reload if the page is accessed via BFcache.
window.addEventListener('pageshow', function (event) {
    if (event.persisted) {
        window.location.reload();
    }
}, { once: true });

// Define the main initialization function for comments.
function initializeCommentsSystem() {
    const commentDataElement = document.getElementById('comment-data');
    if (!commentDataElement) {
        console.warn("Comment data element not found. Comments system might not initialize.");
        return;
    }

    const initialCommentsScript = document.getElementById('initial-comments-json');
    let INITIAL_COMMENTS_DATA = [];

    if (initialCommentsScript && initialCommentsScript.textContent) {
        try {
            const parsedData = JSON.parse(initialCommentsScript.textContent);
            INITIAL_COMMENTS_DATA = Array.isArray(parsedData) ? parsedData : [];
        } catch (e) {
            console.error("Error parsing initial comments JSON:", e);
        }
    } else {
        console.warn("Script tag 'initial-comments-json' not found or empty. Initial comments might be missing.");
    }

    const STUDENT_ID = commentDataElement.dataset.studentId;
    const VIDEO_ID = commentDataElement.dataset.videoId;
    const CSRF_TOKEN = commentDataElement.dataset.csrfToken;
    const CURRENT_USER_NAME = commentDataElement.dataset.currentUserName;

    const ADD_COMMENT_URL = commentDataElement.dataset.addCommentUrl;
    const UPDATE_COMMENT_URL = commentDataElement.dataset.updateCommentUrl;
    const DELETE_COMMENT_URL = commentDataElement.dataset.deleteCommentUrl;
    const EDIT_COMMENT_URL = commentDataElement.dataset.editCommentUrl;
    const ADD_REPLY_URL = commentDataElement.dataset.addReplyUrl;
    // NEW: URLs for replies
    const EDIT_REPLY_URL = commentDataElement.dataset.editReplyUrl;
    const DELETE_REPLY_URL = commentDataElement.dataset.deleteReplyUrl;

    const addCommentBtn = document.getElementById('add-comment-btn');
    const commentAreaContainer = document.getElementById('comment-area-container');
    const draggableCommentsContainer = document.getElementById('draggable-comments-container');
    const commentModal = document.getElementById('comment-modal');
    const commentInput = document.getElementById('comment-input');
    const saveCommentBtn = document.getElementById('save-comment-btn');
    const cancelCommentBtn = document.getElementById('cancel-comment-btn');

    let isCommentingMode = false;
    let newCommentPlacement = { x: 0, y: 0 };
    let commentsData = INITIAL_COMMENTS_DATA;
    let expandedCommentElement = null; // The comment icon div
    let fullContentOverlay = null;     // The full overlay container

    // Flag to indicate if we are editing an existing comment
    let isEditingComment = false;
    let commentToEditId = null;

    // Flags for reply editing/deleting
    let isEditingReply = false;
    let replyToEditId = null;

    // Keep track of the currently open reply dropdown
    let currentOpenReplyDropdown = null;
    // Keep track of the currently open main comment dropdown
    let currentOpenCommentDropdown = null;

    // Variables for the dynamic delete confirmation modal
    let currentDeleteTarget = null; // Can be 'comment' or 'reply'
    let currentCommentIdToDelete = null;
    let currentReplyIdToDelete = null;
    let currentParentCommentIdForReplyDeletion = null;


    // IMPORTANT: draggableCommentsContainer should *always* have pointer-events: auto
    // so that individual comment icons (which are children) can be clicked and hovered.
    if (draggableCommentsContainer) {
        draggableCommentsContainer.style.pointerEvents = 'auto';
    }


    // --- Utility Functions ---

    function truncateText(text, maxLength) {
        if (text && text.length > maxLength) {
            return text.substring(0, maxLength) + '...';
        }
        return text || '';
    }

    /**
     * Auto-resizes a textarea to fit its content.
     * @param {HTMLElement} textarea - The textarea element to resize.
     */
    function autoResizeTextarea(textarea) {
        textarea.style.height = 'auto'; // Reset height to recalculate
        textarea.style.height = (textarea.scrollHeight) + 'px'; // Set height to scroll height
    }

    /**
     * Disables or enables a button based on the text area content.
     * @param {HTMLTextAreaElement} textarea - The textarea to check.
     * @param {HTMLButtonElement} button - The button to enable/disable.
     */
    function toggleSaveButton(textarea, button) {
        if (textarea.value.trim() === '') {
            button.disabled = true;
            button.classList.add('opacity-50', 'cursor-not-allowed');
            button.classList.remove('hover:bg-blue-600'); // Assuming blue is default hover
        } else {
            button.disabled = false;
            button.classList.remove('opacity-50', 'cursor-not-allowed');
            button.classList.add('hover:bg-blue-600'); // Assuming blue is default hover
        }
    }


    /**
     * Closes any currently expanded comment overlay.
     */
    function closeExpandedComment() {
        // Close any open reply dropdown first
        if (currentOpenReplyDropdown) {
            currentOpenReplyDropdown.remove();
            currentOpenReplyDropdown = null;
        }
        // Close any open main comment dropdown
        if (currentOpenCommentDropdown) {
            currentOpenCommentDropdown.remove();
            currentOpenCommentDropdown = null;
        }

        if (fullContentOverlay) {
            // If editing a reply, cancel edit mode before closing
            const overlayToClose = fullContentOverlay;
            const iconToReset = expandedCommentElement;

            // If editing a reply, cancel edit mode before closing
            if (isEditingReply) {
                const currentReplyElement = overlayToClose.querySelector(`[data-reply-id="${replyToEditId}"]`);
                if (currentReplyElement) {
                    const replyTextEl = currentReplyElement.querySelector('.reply-text');
                    const replyEditInput = currentReplyElement.querySelector('.edit-reply-input');
                    const replyActions = currentReplyElement.querySelector('.edit-reply-actions');

                    if (replyTextEl && replyEditInput && replyActions) {
                        replyTextEl.classList.remove('hidden');
                        replyEditInput.classList.add('hidden');
                        replyActions.classList.add('hidden');
                    }
                }
                isEditingReply = false;
                replyToEditId = null;
            }
            // If editing a main comment, cancel edit mode before closing
            if (isEditingComment) {
                const commentTextElement = overlayToClose.querySelector(`#comment-text-${commentToEditId}`);
                const editInput = overlayToClose.querySelector('.edit-comment-input');
                const editActions = overlayToClose.querySelector('.edit-comment-actions');

                if (commentTextElement && editInput && editActions) {
                    commentTextElement.classList.remove('hidden');
                    editInput.classList.add('hidden');
                    editActions.classList.add('hidden');
                }
                isEditingComment = false;
                commentToEditId = null;
            }

            overlayToClose.classList.remove('opacity-100', 'scale-100', 'pointer-events-auto');
            overlayToClose.classList.add('opacity-0', 'scale-95', 'pointer-events-none', 'invisible');

            if (iconToReset) {
                iconToReset.classList.remove('is-expanded');
                makeDraggable(iconToReset, iconToReset.id, true);
                iconToReset.style.cursor = 'grab'; // Reset cursor
                iconToReset.style.zIndex = '30'; // Revert icon z-index
            }

            // Remove the captured overlay from DOM after transition completes.
            setTimeout(() => {
                if (overlayToClose && overlayToClose.parentNode) {
                    overlayToClose.remove();
                }
            }, 300); // Match this with your CSS transition duration

            fullContentOverlay = null;
            expandedCommentElement = null;
        }
    }

    async function sendAjaxRequest(url, method, data) {
        try {
            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': CSRF_TOKEN
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                let errorData = {};
                try {
                    errorData = await response.json();
                } catch (e) {
                    errorData.message = response.statusText || 'Unknown error. Response not JSON.';
                }
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error("AJAX request failed:", error);
            throw error;
        }
    }

    /**
     * Creates an HTML element for a single reply.
     * @param {object} reply - The reply object.
     * @param {string} parentCommentId - The ID of the parent comment.
     * @returns {HTMLElement} The created reply element.
     */
    function createReplyElement(reply, parentCommentId) {
        const replyDiv = document.createElement('div');
        replyDiv.className = 'flex items-start mb-4';
        replyDiv.dataset.replyId = reply.id;

        const isAuthor = reply.CommentedBy === CURRENT_USER_NAME;

        replyDiv.innerHTML = `
            <div class="w-10 h-10 bg-gray-300 text-gray-800 rounded-full flex items-center justify-center text-md font-bold mr-3 flex-shrink-0">
                ${reply.CommentedBy ? reply.CommentedBy.charAt(0).toUpperCase() : 'U'}
            </div>
            <div class="flex-grow">
                <div class="flex items-start justify-between w-full mb-1">
                    <div class="flex flex-col min-w-0">
                        <p class="font-semibold text-base">${reply.CommentedBy || 'Unknown User'}</p>
                        <p class="text-xs text-gray-500 mb-2">${reply.FormattedDate || 'No Date'}</p> </div>
                    ${isAuthor ? `
                    <div class="relative flex-shrink-0 flex items-center">
                        <button class="reply-options-btn p-1 rounded-full hover:bg-gray-100 focus:outline-none">
                            <svg class="w-6 h-6 text-gray-500 hover:text-gray-700" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                                <path d="M2 10a2 2 0 114 0a2 2 0 01-4 0zM8 10a2 2 0 114 0a2 2 0 01-4 0zM14 10a2 2 0 114 0a2 2 0 01-4 0z"></path>
                            </svg>
                        </button>
                        </div>
                    ` : ''}
                </div>
                <p class="text-sm break-words overflow-hidden reply-text" id="reply-text-${reply.id}" style="white-space: pre-wrap; word-break: break-word;">${reply.Comment}</p>
                <textarea class="edit-reply-input w-full p-2 border border-gray-300 rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500 hidden" rows="4"></textarea> <div class="edit-reply-actions flex justify-end mb-3 flex-shrink-0 hidden w-full">
                    <button class="cancel-edit-reply-btn bg-gray-300 text-gray-800 px-3 py-2 text-sm rounded-md mr-2 hover:bg-gray-400">Cancel</button>
                    <button class="save-edit-reply-btn bg-blue-500 text-white px-3 py-2 text-sm rounded-md hover:bg-blue-600">Save</button>
                </div>
            </div>
        `;

        // Add event listeners for reply options
        if (isAuthor) {
            const optionsBtn = replyDiv.querySelector('.reply-options-btn');
            const replyOptionsContainer = replyDiv.querySelector('.reply-options-container');

            // Create the dropdown element
            const optionsDropdown = document.createElement('div');
            optionsDropdown.className = 'reply-options-dropdown absolute w-40 bg-white rounded-md shadow-lg py-1 z-[150] hidden'; // Higher z-index
            optionsDropdown.setAttribute('data-parent-reply-id', reply.id); // Link to its reply
            optionsDropdown.innerHTML = `
                <button class="edit-reply-option block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100" data-reply-id="${reply.id}">Edit</button>
                <button class="delete-reply-option block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100" data-reply-id="${reply.id}">Delete</button>
            `;

            const editReplyBtn = optionsDropdown.querySelector('.edit-reply-option');
            const deleteReplyBtn = optionsDropdown.querySelector('.delete-reply-option');
            const saveEditReplyBtn = replyDiv.querySelector('.save-edit-reply-btn');
            const cancelEditReplyBtn = replyDiv.querySelector('.cancel-edit-reply-btn');
            const editReplyInput = replyDiv.querySelector('.edit-reply-input');


            if (optionsBtn) {
                optionsBtn.addEventListener('click', (e) => {
                    e.stopPropagation();

                    // Close any other open reply dropdown
                    if (currentOpenReplyDropdown && currentOpenReplyDropdown !== optionsDropdown) {
                        currentOpenReplyDropdown.remove();
                    }
                    // Close main comment dropdown if open
                    if (currentOpenCommentDropdown) {
                        currentOpenCommentDropdown.remove();
                        currentOpenCommentDropdown = null;
                    }


                    if (optionsDropdown.classList.contains('hidden')) {
                        // Position the dropdown dynamically
                        const btnRect = optionsBtn.getBoundingClientRect();
                        // Position relative to the draggableCommentsContainer (the video area)
                        const containerRect = draggableCommentsContainer.getBoundingClientRect();

                        optionsDropdown.style.top = `${btnRect.top - containerRect.top + btnRect.height + 5}px`; // 5px offset
                        optionsDropdown.style.left = `${btnRect.left - containerRect.left + btnRect.width - optionsDropdown.offsetWidth}px`; // Align right edge
                        optionsDropdown.style.position = 'absolute'; // Ensure it's absolutely positioned within the container
                        optionsDropdown.classList.remove('hidden');
                        draggableCommentsContainer.appendChild(optionsDropdown); // Append to the main draggable container
                        currentOpenReplyDropdown = optionsDropdown;
                    } else {
                        optionsDropdown.classList.add('hidden');
                        optionsDropdown.remove(); // Remove from DOM when hidden
                        currentOpenReplyDropdown = null;
                    }
                });
            }

            if (editReplyBtn) {
                editReplyBtn.addEventListener('click', () => {
                    // This closeExpandedComment here might be too aggressive if we want to edit replies within the open popup
                    // Consider if a more granular close is needed, or if this function should only close the *entire* comment popup
                    // closeExpandedComment();
                    const replyTextEl = replyDiv.querySelector('.reply-text');
                    const editInput = replyDiv.querySelector('.edit-reply-input');
                    const editActions = replyDiv.querySelector('.edit-reply-actions');

                    replyTextEl.classList.add('hidden');
                    editInput.value = reply.Comment;
                    editInput.classList.remove('hidden');
                    editActions.classList.remove('hidden');
                    if (currentOpenReplyDropdown) { // Hide and remove the dynamically added dropdown
                        currentOpenReplyDropdown.classList.add('hidden');
                        currentOpenReplyDropdown.remove();
                        currentOpenReplyDropdown = null;
                    }
                    editInput.focus();

                    isEditingReply = true;
                    replyToEditId = reply.id;

                    // Immediately check and toggle save button on showing edit mode
                    toggleSaveButton(editReplyInput, saveEditReplyBtn);
                });
            }

            // Listen for input changes in the reply edit textarea
            if (editReplyInput && saveEditReplyBtn) {
                editReplyInput.addEventListener('input', () => {
                    toggleSaveButton(editReplyInput, saveEditReplyBtn);
                });
            }


            if (deleteReplyBtn) {
                deleteReplyBtn.addEventListener('click', () => {
                    if (currentOpenReplyDropdown) { // Hide and remove the dynamically added dropdown
                        currentOpenReplyDropdown.classList.add('hidden');
                        currentOpenReplyDropdown.remove();
                        currentOpenReplyDropdown = null;
                    }
                    // Changed to use the unified modal
                    showConfirmDeleteModal('reply', reply.id, parentCommentId);
                });
            }

            if (saveEditReplyBtn) {
                saveEditReplyBtn.addEventListener('click', async () => {
                    const newReplyText = replyDiv.querySelector('.edit-reply-input').value.trim();
                    if (newReplyText && replyToEditId) {
                        await handleEditReply(replyToEditId, parentCommentId, newReplyText);
                        // After successful edit, revert UI
                        const replyTextEl = replyDiv.querySelector('.reply-text');
                        const editInput = replyDiv.querySelector('.edit-reply-input');
                        const editActions = replyDiv.querySelector('.edit-reply-actions');

                        replyTextEl.textContent = newReplyText;
                        replyTextEl.classList.remove('hidden');
                        editInput.classList.add('hidden');
                        editActions.classList.add('hidden');

                        isEditingReply = false;
                        replyToEditId = null;
                        toggleSaveButton(editReplyInput, saveEditReplyBtn); // Re-check button state (should be disabled if hidden)
                    } else {
                        // This alert is now replaced by the button disabling, but leaving as console.log for debugging
                        console.log("Edited reply cannot be empty!");
                    }
                });
            }

            if (cancelEditReplyBtn) {
                cancelEditReplyBtn.addEventListener('click', () => {
                    const replyTextEl = replyDiv.querySelector('.reply-text');
                    const editInput = replyDiv.querySelector('.edit-reply-input');
                    const editActions = replyDiv.querySelector('.edit-reply-actions');

                    replyTextEl.classList.remove('hidden');
                    editInput.classList.add('hidden');
                    editActions.classList.add('hidden');

                    isEditingReply = false;
                    replyToEditId = null;
                    toggleSaveButton(editReplyInput, saveEditReplyBtn); // Re-check button state (should be disabled if hidden)
                });
            }

            // No longer need replyDiv.addEventListener('click') for closing dropdown as it's global
        }
        return replyDiv;
    }

    /**
        * Creates and appends a draggable comment icon with hover preview and click-to-expand functionality.
        * @param {object} comment - The comment object {id, Comment, CommentedBy, FormattedDate, x_pos, y_pos, replies}.
        */
    function createCommentElement(comment) {
        const x_pos = parseFloat(comment.x_pos);
        const y_pos = parseFloat(comment.y_pos);

        if (isNaN(x_pos) || isNaN(y_pos)) {
            return null; // Return null if comment is not position-based
        }

        const isCommentAuthor = comment.CommentedBy === CURRENT_USER_NAME;

        // 1. The main comment container (now the visual icon with solid black border)
        const commentContainerDiv = document.createElement('div');
        commentContainerDiv.id = comment.id;
        commentContainerDiv.className = 'absolute z-30 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold border-2 border-black hover:bg-blue-600 transition-colors duration-200 focus:outline-none focus:ring-0 cursor-grab';
        // This positions the icon's center directly at (x_pos, y_pos).
        commentContainerDiv.style.left = `${x_pos}px`;
        commentContainerDiv.style.top  = `${y_pos}px`;
        commentContainerDiv.style.transform = 'translate(-50%, -50%)';

        // The SVG element directly inside the commentContainerDiv
        commentContainerDiv.innerHTML = `<svg class="w-6 h-6" fill="white" stroke="black" stroke-width="1.5" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.336-3.11c-.813-1.013-1.336-2.31-1.336-3.89C2 6.134 5.582 3 10 3s8 3.134 8 7z" clip-rule="evenodd"></path></svg>`;

        // 2. Preview content (hidden by default, shown on hover)
        const previewContent = document.createElement('div');
        previewContent.className = 'absolute top-full left-1/2 -translate-x-1/2 mt-2 p-2 bg-white rounded-lg shadow-lg text-xs text-gray-800 opacity-0 scale-0 origin-top transition-all duration-200 ease-out pointer-events-none w-48 max-w-xs overflow-hidden invisible';

        // Calculate reply count
        const replyCount = comment.replies ? comment.replies.length : 0;
        const replyText = replyCount === 1 ? '1 reply' : `${replyCount} replies`;

        previewContent.innerHTML = `
            <p class="font-semibold mb-1 flex items-baseline">
                <span>${comment.CommentedBy || 'Unknown User'}</span>
                <span class="text-gray-500 text-xs ml-2">${comment.FormattedDate || 'No Date'}</span>
            </p>
            <p style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${truncateText(comment.Comment, 50)}</p>
            <p class="text-gray-600 mt-1">${replyText}</p> `;
        commentContainerDiv.appendChild(previewContent);

        // 3. Full content overlay (hidden by default, shown on click)
        const fullContentOverlayElement = document.createElement('div');
        // Changed from fixed modal to absolute popup
        fullContentOverlayElement.className = 'absolute bg-white rounded-lg shadow-2xl p-6 w-96 transform transition-all duration-300 ease-out opacity-0 scale-95 pointer-events-none invisible z-[100] max-h-[90vh] flex flex-col overflow-hidden';
        fullContentOverlayElement.style.minWidth = '300px'; // Ensure a reasonable minimum width for the popup
        fullContentOverlayElement.style.maxWidth = '400px'; // Max width for the popup

        const fullContentCard = document.createElement('div');
        fullContentCard.className = 'flex flex-col flex-grow'; // Ensure the card itself allows vertical growth if needed
        fullContentCard.innerHTML = `
            <div class="relative flex items-center justify-end w-full mb-1">
            ${isCommentAuthor ? `
                <button class="delete-all-options-btn p-1 rounded-full hover:bg-gray-100 focus:outline-none mr-2">
                    <svg class="w-6 h-6 text-gray-500 hover:text-gray-700" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 10a2 2 0 114 0a2 2 0 01-4 0zM8 10a2 2 0 114 0a2 2 0 01-4 0zM14 10a2 2 0 114 0a2 2 0 01-4 0z"></path>
                    </svg>
                </button>
                ` : ''}
                <button class="close-comment-btn p-1 rounded-full hover:bg-gray-100 focus:outline-none">
                    <svg class="w-6 h-6 text-gray-500 hover:text-gray-700" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </button>
                <div class="delete-all-dropdown absolute right-0 top-full mt-2 w-40 bg-white rounded-md shadow-lg py-1 z-20 hidden">
                    <button class="delete-entire-comment-option block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Delete Comment</button>
                </div>
            </div>

            <hr class="my-2 mb-3 border-gray-300">

            <div class="flex items-start mb-4">
                <div class="w-10 h-10 bg-gray-300 text-gray-800 rounded-full flex items-center justify-center text-md font-bold mr-3">
                    ${comment.CommentedBy ? comment.CommentedBy.charAt(0).toUpperCase() : 'U'}
                </div>
                <div class="flex-grow">
                    <div class="flex items-start justify-between w-full">
                        <div class="flex flex-col">
                            <p class="font-semibold text-base">${comment.CommentedBy || 'Unknown User'}</p>
                            <p class="text-xs text-gray-500 mb-2">${comment.FormattedDate || 'No Date'}</p>
                        </div>
                        ${isCommentAuthor ? `
                        <div class="relative flex items-center">
                            <button class="comment-options-btn p-1 rounded-full hover:bg-gray-100 focus:outline-none">
                                <svg class="w-6 h-6 text-gray-500 hover:text-gray-700" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M2 10a2 2 0 114 0a2 2 0 01-4 0zM8 10a2 2 0 114 0a2 2 0 01-4 0zM14 10a2 2 0 114 0a2 2 0 01-4 0z"></path>
                                </svg>
                            </button>
                            </div>
                        ` : ''}
                    </div>
                    <p class="text-sm flex-shrink-0" id="comment-text-${comment.id}" style="white-space: pre-wrap; word-break: break-word;">${comment.Comment}</p>
                    <textarea class="edit-comment-input w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 hidden" rows="4"></textarea>
                </div>
            </div>
            <div class="edit-comment-actions flex justify-end mb-3 flex-shrink-0 hidden">
                <button class="cancel-edit-btn bg-gray-300 text-gray-800 px-3 py-2 text-sm rounded-md mr-2 hover:bg-gray-400">Cancel</button>
                <button class="save-edit-btn bg-blue-500 text-white px-3 py-2 text-sm rounded-md hover:bg-blue-600">Save</button>
            </div>
            <div class="replies-container overflow-y-auto w-full mb-4 flex-grow">
            </div>

            <div class="flex items-center mb-3 flex-shrink-0">
                <textarea placeholder="Add a reply..." class="reply-input flex-grow p-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" rows="1" style="min-height: 40px; resize: none; overflow-y: hidden;"></textarea>
                <button class="reply-btn ml-2 bg-blue-500 text-white px-3 py-2 rounded-md transition-colors duration-200 flex items-center justify-center">
                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                    </svg>
                </button>
            </div>
        `;
        fullContentOverlayElement.appendChild(fullContentCard);


        // Create the main comment options dropdown element
        const commentOptionsBtn = fullContentCard.querySelector('.comment-options-btn');
        const commentOptionsDropdown = document.createElement('div');
        commentOptionsDropdown.className = 'comment-options-dropdown absolute w-40 bg-white rounded-md shadow-lg py-1 z-[150] hidden'; // Higher z-index for main comment dropdown
        commentOptionsDropdown.setAttribute('data-parent-comment-id', comment.id); // Link to its comment
        commentOptionsDropdown.innerHTML = `
            <button class="edit-comment-only-option block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100" data-comment-id="${comment.id}">Edit</button>
        `;
        // Do NOT append it to fullContentCard yet. It will be appended dynamically.


        // Populate replies if they exist
        const repliesContainer = fullContentCard.querySelector('.replies-container');
        if (comment.replies && Array.isArray(comment.replies) && repliesContainer) {
            comment.replies.forEach(reply => {
                // Pass parentCommentId to createReplyElement
                const replyEl = createReplyElement(reply, comment.id);
                if (replyEl) {
                    repliesContainer.appendChild(replyEl);
                }
            });
        }

        // Get the edit elements for the main comment
        const editCommentInput = fullContentOverlayElement.querySelector('.edit-comment-input');
        const saveEditCommentBtn = fullContentOverlayElement.querySelector('.save-edit-btn');


        // --- Event Listeners for the Icon (commentContainerDiv) ---
        commentContainerDiv.addEventListener('mouseenter', () => {
            if (!commentContainerDiv.classList.contains('is-expanded')) {
                previewContent.classList.remove('opacity-0', 'scale-0', 'invisible');
                previewContent.classList.add('opacity-100', 'scale-100');
                commentContainerDiv.style.zIndex = '40';
            }
        });

        commentContainerDiv.addEventListener('mouseleave', () => {
            if (!commentContainerDiv.classList.contains('is-expanded')) {
                previewContent.classList.remove('opacity-100', 'scale-100');
                previewContent.classList.add('opacity-0', 'scale-0', 'invisible');
                commentContainerDiv.style.zIndex = '30';
            }
        });

        commentContainerDiv.addEventListener('click', (e) => {
            e.stopPropagation();

            // Prevent opening modal if clicking inside the modal or on a button within it
            if (e.target.closest('.full-content-overlay-element') || // This class should ideally be on the overlay itself
                e.target.closest('.delete-all-options-btn') ||
                e.target.closest('.delete-all-dropdown') ||
                e.target.closest('.delete-entire-comment-option') ||
                e.target.closest('.close-comment-btn') ||
                e.target.closest('.reply-btn') ||
                e.target.closest('.reply-input') ||
                e.target.closest('.save-edit-btn') ||
                e.target.closest('.cancel-edit-btn') ||
                e.target.closest('.comment-options-btn') ||
                e.target.closest('.comment-options-dropdown') ||
                e.target.closest('.edit-comment-only-option') ||
                e.target.closest('.delete-comment-only-option') ||
                // Added reply-specific buttons to stop propagation
                e.target.closest('.reply-options-btn') ||
                e.target.closest('.reply-options-dropdown') ||
                e.target.closest('.edit-reply-option') ||
                e.target.closest('.delete-reply-option') ||
                e.target.closest('.save-edit-reply-btn') ||
                e.target.closest('.cancel-edit-reply-btn')
            ) {
                return;
            }

            if (!commentContainerDiv.classList.contains('is-expanded')) {
                closeExpandedComment(); // Close any other open comment

                expandedCommentElement = commentContainerDiv;
                fullContentOverlay = fullContentOverlayElement; // Assign the new overlay element

                draggableCommentsContainer.appendChild(fullContentOverlay); // Append to the draggable container

                // Position the popup
                const iconRect = commentContainerDiv.getBoundingClientRect();
                const containerRect = draggableCommentsContainer.getBoundingClientRect();

                let popupLeft = iconRect.left - containerRect.left + iconRect.width + 10; // 10px right of icon
                let popupTop = iconRect.top - containerRect.top;

                // Adjust if it goes off screen right
                if (popupLeft + fullContentOverlay.offsetWidth > containerRect.width) {
                    popupLeft = iconRect.left - containerRect.left - fullContentOverlay.offsetWidth - 10; // 10px left of icon
                }
                // Adjust if it goes off screen bottom
                if (popupTop + fullContentOverlay.offsetHeight > containerRect.height) {
                    popupTop = containerRect.height - fullContentOverlay.offsetHeight - 10; // Align to bottom of container
                }
                // Ensure it doesn't go off screen top
                if (popupTop < 0) {
                    popupTop = 10; // 10px from top of container
                }

                fullContentOverlay.style.left = `${popupLeft}px`;
                fullContentOverlay.style.top = `${popupTop}px`;


                fullContentOverlay.classList.remove('opacity-0', 'scale-95', 'pointer-events-none', 'invisible');
                fullContentOverlay.classList.add('opacity-100', 'scale-100', 'pointer-events-auto');

                commentContainerDiv.classList.add('is-expanded');
                commentContainerDiv.style.zIndex = '50'; // Bring the icon to front when its modal is open

                previewContent.classList.remove('opacity-100', 'scale-100');
                previewContent.classList.add('opacity-0', 'scale-0', 'invisible');

                makeDraggable(commentContainerDiv, comment.id, false); // Disable dragging when expanded
                commentContainerDiv.style.cursor = 'default';
            }
        });


        // --- Event Listeners for the Full Content Overlay ---
        fullContentOverlayElement.querySelector('.close-comment-btn').addEventListener('click', () => {
            closeExpandedComment();
        });

        // Toggle options dropdown for main comment (Delete Entire Comment)
        const deleteAllOptionsBtn = fullContentOverlayElement.querySelector('.delete-all-options-btn');
        const deleteAllDropdown = fullContentOverlayElement.querySelector('.delete-all-dropdown');
        if (deleteAllOptionsBtn) {
            deleteAllOptionsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (deleteAllDropdown) deleteAllDropdown.classList.toggle('hidden');
            });
        }


        // Handle Delete Entire Comment option (main comment from dropdown)
        const deleteEntireCommentOption = fullContentOverlayElement.querySelector('.delete-entire-comment-option');
        if (deleteEntireCommentOption) {
            deleteEntireCommentOption.addEventListener('click', (event) => {
                event.stopPropagation();
                if (deleteAllDropdown) deleteAllDropdown.classList.add('hidden'); // Hide dropdown immediately
                // Modified to use the unified modal
                showConfirmDeleteModal('comment', comment.id);
            });
        }


        // NEW: Toggle options dropdown for comment-specific actions (Edit/Delete Comment only)
        if (commentOptionsBtn) {
            commentOptionsBtn.addEventListener('click', (e) => {
                e.stopPropagation();

                // Close any other open comment dropdown
                if (currentOpenCommentDropdown && currentOpenCommentDropdown !== commentOptionsDropdown) {
                    currentOpenCommentDropdown.remove();
                }
                // Close any open reply dropdown
                if (currentOpenReplyDropdown) {
                    currentOpenReplyDropdown.remove();
                    currentOpenReplyDropdown = null;
                }

                if (commentOptionsDropdown.classList.contains('hidden')) {
                    const btnRect = commentOptionsBtn.getBoundingClientRect();
                    // Position relative to the draggableCommentsContainer (the video area)
                    const containerRect = draggableCommentsContainer.getBoundingClientRect();

                    commentOptionsDropdown.style.top = `${btnRect.top - containerRect.top + btnRect.height + 5}px`;
                    commentOptionsDropdown.style.left = `${btnRect.left - containerRect.left + btnRect.width - commentOptionsDropdown.offsetWidth}px`;
                    commentOptionsDropdown.style.position = 'absolute';
                    commentOptionsDropdown.classList.remove('hidden');
                    draggableCommentsContainer.appendChild(commentOptionsDropdown); // Append to the main draggable container
                    currentOpenCommentDropdown = commentOptionsDropdown;
                } else {
                    commentOptionsDropdown.classList.add('hidden');
                    commentOptionsDropdown.remove();
                    currentOpenCommentDropdown = null;
                }
            });
        }

        // NEW: Handle Edit Comment (only) option
        const editCommentOnlyOption = commentOptionsDropdown.querySelector('.edit-comment-only-option');
        if (editCommentOnlyOption) {
            editCommentOnlyOption.addEventListener('click', () => {
                // closeExpandedComment(); // This might close the entire popup, reconsider if necessary
                const commentTextElement = fullContentOverlayElement.querySelector(`#comment-text-${comment.id}`);
                const editInput = fullContentOverlayElement.querySelector('.edit-comment-input');
                const editActions = fullContentOverlayElement.querySelector('.edit-comment-actions');

                // Hide all dropdowns
                if (deleteAllDropdown) deleteAllDropdown.classList.add('hidden');
                if (currentOpenCommentDropdown) { // Hide and remove the dynamically added dropdown
                    currentOpenCommentDropdown.classList.add('hidden');
                    currentOpenCommentDropdown.remove();
                    currentOpenCommentDropdown = null;
                }

                commentTextElement.classList.add('hidden');
                editInput.value = comment.Comment;
                editInput.classList.remove('hidden');
                editActions.classList.remove('hidden');
                editInput.focus();

                isEditingComment = true;
                commentToEditId = comment.id;

                // Immediately check and toggle save button on showing edit mode
                toggleSaveButton(editCommentInput, saveEditCommentBtn);
            });
        }

        // Listen for input changes in the comment edit textarea
        if (editCommentInput && saveEditCommentBtn) {
            editCommentInput.addEventListener('input', () => {
                toggleSaveButton(editCommentInput, saveEditCommentBtn);
            });
        }


        // NEW: Handle Delete Comment (only) option
        // The original code does not have a "Delete Comment (only)" option in the main comment dropdown.
        // It only has "Delete Entire Comment" (which deletes comment + replies).
        // If you intend to add a "Delete Comment (only)" feature that leaves replies,
        // you would add a button here and a corresponding handleDeleteOnlyComment function.
        // For now, I'm keeping the original logic which only has the "Delete Entire Comment" option.
        const deleteCommentOnlyOption = commentOptionsDropdown.querySelector('.delete-comment-only-option');
        if (deleteCommentOnlyOption) {
            deleteCommentOnlyOption.addEventListener('click', (event) => {
                event.stopPropagation();
                if (currentOpenCommentDropdown) { // Hide and remove the dynamically added dropdown
                    currentOpenCommentDropdown.classList.add('hidden');
                    currentOpenCommentDropdown.remove();
                    currentOpenCommentDropdown = null;
                }
                // If you implement handleDeleteOnlyComment, call it here.
                // handleDeleteOnlyComment(comment.id);
            });
        }

        // Handle Save Edit (main comment)
        if (saveEditCommentBtn) { // Added check for saveEditCommentBtn
            saveEditCommentBtn.addEventListener('click', async () => {
                const editInput = fullContentOverlayElement.querySelector('.edit-comment-input');
                const newCommentText = editInput.value.trim();

                if (newCommentText && commentToEditId) {
                    await handleEditComment(commentToEditId, newCommentText);
                    // After successful edit, revert UI
                    const commentTextElement = fullContentOverlayElement.querySelector(`#comment-text-${comment.id}`);
                    const editActions = fullContentOverlayElement.querySelector('.edit-comment-actions');

                    commentTextElement.textContent = newCommentText;
                    commentTextElement.classList.remove('hidden');
                    editInput.classList.add('hidden');
                    editActions.classList.add('hidden');

                    isEditingComment = false;
                    commentToEditId = null;
                    toggleSaveButton(editCommentInput, saveEditCommentBtn); // Re-check button state (should be disabled if hidden)

                    // Also update the preview content for the comment icon
                    const currentComment = commentsData.find(c => c.id === comment.id);
                    if (currentComment) {
                        currentComment.Comment = newCommentText;
                        // Update reply count in preview as well if replies can be added/deleted live
                        const updatedReplyCount = currentComment.replies ? currentComment.replies.length : 0;
                        const updatedReplyText = updatedReplyCount === 1 ? '1 reply' : `${updatedReplyCount} replies`;

                        previewContent.innerHTML = `
                            <p class="font-semibold mb-1">${currentComment.CommentedBy || 'Unknown User'}</p>
                            <p style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${truncateText(currentComment.Comment, 50)}</p>
                            <p class="text-gray-600 mt-1">${updatedReplyText}</p> `;
                    }

                } else {
                    console.log("Edited comment cannot be empty!"); // Changed from alert()
                }
            });
        }


        // Handle Cancel Edit (main comment)
        fullContentOverlayElement.querySelector('.cancel-edit-btn').addEventListener('click', () => {
            const commentTextElement = fullContentOverlayElement.querySelector(`#comment-text-${comment.id}`);
            const editInput = fullContentOverlayElement.querySelector('.edit-comment-input');
            const editActions = fullContentOverlayElement.querySelector('.edit-comment-actions');

            commentTextElement.classList.remove('hidden');
            editInput.classList.add('hidden');
            editActions.classList.add('hidden');

            isEditingComment = false;
            commentToEditId = null;
            toggleSaveButton(editCommentInput, saveEditCommentBtn); // Re-check button state (should be disabled if hidden)
        });

        fullContentOverlayElement.querySelector('.reply-btn').addEventListener('click', async () => {
            const replyInput = fullContentOverlayElement.querySelector('.reply-input');
            const replyText = replyInput.value.trim();
            if (replyText) {
                await handleAddReply(comment.id, replyText, repliesContainer);
                replyInput.value = '';
                replyInput.style.height = 'auto'; // Reset height after sending reply

                // IMPORTANT: After adding a new reply, update the preview content's reply count
                const currentComment = commentsData.find(c => c.id === comment.id);
                if (currentComment) {
                    const updatedReplyCount = currentComment.replies ? currentComment.replies.length : 0;
                    const updatedReplyText = updatedReplyCount === 1 ? '1 reply' : `${updatedReplyCount} replies`;
                    // Assuming previewContent is still in scope, which it should be
                    previewContent.querySelector('p:last-child').textContent = updatedReplyText; // Target the last paragraph for the update
                }
            } else {
                console.log("Reply cannot be empty!"); // Changed from alert()
            }
        });

        // NEW: Add event listener for auto-resizing the reply input textarea
        const replyInputTextarea = fullContentOverlayElement.querySelector('.reply-input');
        if (replyInputTextarea) {
            replyInputTextarea.addEventListener('input', () => {
                autoResizeTextarea(replyInputTextarea);
            });
        }

        // Close main comment dropdown if clicked outside of it (within the overlay but not on dropdown/button)
        // This behavior might need adjustment if the popup itself should not close on outside click *within the container*
        // but only if clicked on the general document background.
        // For now, the global listener handles closing.
        fullContentOverlayElement.addEventListener('click', (e) => {
            // No longer check e.target === fullContentOverlayElement for closing the popup,
            // as the popup doesn't have a distinct background now.
            // Closing is now handled by the global document click listener below.

            // Close main comment options dropdown if clicked outside of it
            if (deleteAllDropdown && !deleteAllDropdown.classList.contains('hidden') && deleteAllOptionsBtn && !deleteAllOptionsBtn.contains(e.target) && !deleteAllDropdown.contains(e.target)) {
                deleteAllDropdown.classList.add('hidden');
            }
            // Comment and reply dropdowns are handled by global listener
        });

        // Ensure draggableCommentsContainer is not null before appending
        if (draggableCommentsContainer) {
            draggableCommentsContainer.appendChild(commentContainerDiv);
            makeDraggable(commentContainerDiv, comment.id, true); // Enable dragging initially
        }
        return commentContainerDiv;
    }
    /**
     * Makes an element draggable within its parent container using a transform-consistent method.
     * @param {HTMLElement} element - The element to make draggable.
     * @param {string} commentId - The ID of the comment associated with this element.
     * @param {boolean} enable - True to enable dragging, false to disable.
     */
    function makeDraggable(element, commentId, enable) {
        // Clean up previous listeners if they exist to prevent duplicates
        if (element._dragMouseDownHandler) {
            element.removeEventListener('mousedown', element._dragMouseDownHandler);
            // Mousemove and mouseup are on the document, but we'll ensure they are cleaned up
            // by removing them in the mouseup handler itself.
        }
        // Ensure old handlers from the element object are cleared
        delete element._dragMouseDownHandler;
        delete element._dragMouseMoveHandler;
        delete element._dragMouseUpHandler;


        if (enable) {
            let isDraggingElement = false;
            let initialMouseX, initialMouseY; // Mouse position on mousedown
            let initialElementLeft, initialElementTop; // Element's left/top on mousedown

            element._dragMouseDownHandler = (e) => {
                // Only start drag if left mouse button, not on a button, and not expanded
                if (e.button === 0 && !e.target.closest('button') && !element.classList.contains('is-expanded')) {
                    isDraggingElement = true;
                    element.style.cursor = 'grabbing';
                    element.style.zIndex = '45'; // Bring to front while dragging

                    // Store initial mouse position
                    initialMouseX = e.clientX;
                    initialMouseY = e.clientY;

                    // Store initial element position (which are the center coordinates due to transform)
                    initialElementLeft = parseFloat(element.style.left) || 0;
                    initialElementTop = parseFloat(element.style.top) || 0;

                    // Add listeners to the document to handle dragging outside the element
                    document.addEventListener('mousemove', element._dragMouseMoveHandler);
                    document.addEventListener('mouseup', element._dragMouseUpHandler);

                    e.preventDefault(); // Prevent text selection during drag
                    e.stopPropagation();
                }
            };

            element._dragMouseMoveHandler = (e) => {
                if (!isDraggingElement) return;

                // Calculate mouse movement delta
                const deltaX = e.clientX - initialMouseX;
                const deltaY = e.clientY - initialMouseY;

                // Calculate new position for the element's center
                let newX = initialElementLeft + deltaX;
                let newY = initialElementTop + deltaY;

                // Constrain within parent bounds
                if (draggableCommentsContainer) {
                    const parentRect = draggableCommentsContainer.getBoundingClientRect();
                    const elementWidth = element.offsetWidth;
                    const elementHeight = element.offsetHeight;

                    // Since left/top is the center, bounds are offset by half the element's size
                    const minX = elementWidth / 2;
                    const maxX = parentRect.width - (elementWidth / 2);
                    const minY = elementHeight / 2;
                    const maxY = parentRect.height - (elementHeight / 2);

                    newX = Math.max(minX, Math.min(newX, maxX));
                    newY = Math.max(minY, Math.min(newY, maxY));
                }

                element.style.left = `${newX}px`;
                element.style.top = `${newY}px`;
            };

            element._dragMouseUpHandler = () => {
                if (!isDraggingElement) return;
                isDraggingElement = false;
                element.style.cursor = 'grab'; // Reset cursor
                element.style.zIndex = '30'; // Revert z-index

                // The final left/top are the new center coordinates
                const finalX = parseFloat(element.style.left);
                const finalY = parseFloat(element.style.top);

                handleUpdateCommentPosition(commentId, finalX, finalY);

                // Clean up global listeners
                document.removeEventListener('mousemove', element._dragMouseMoveHandler);
                document.removeEventListener('mouseup', element._dragMouseUpHandler);
            };

            element.addEventListener('mousedown', element._dragMouseDownHandler);

        } else {
            // If disabling, ensure cursor is default and no handlers are attached.
            element.style.cursor = 'default';
        }
    }

    /**
     * Shows the comment input modal.
     */
    function showCommentModal() {
        if (commentInput && commentModal) {
            commentInput.value = '';
            commentModal.classList.remove('hidden');
            commentInput.focus();
            // Ensure save button is disabled when modal opens if input is empty
            toggleSaveButton(commentInput, saveCommentBtn);
        }
    }

    /**
     * Hides the comment input modal.
     */
    function hideCommentModal() {
        if (commentModal) {
            commentModal.classList.add('hidden');
        }
    }

    async function handleAddComment(commentText, x_pos, y_pos) {
        const url = ADD_COMMENT_URL;
        try {
            const response = await sendAjaxRequest(url, 'POST', { comment: commentText, x_pos: x_pos, y_pos: y_pos });
            if (response.comment) {
                commentsData.push(response.comment);
                const newCommentEl = createCommentElement(response.comment); // The createCommentElement already appends it if it's a positional comment.
            }
        } catch (error) {
            console.log('Failed to add comment: ' + error.message); // Changed from alert()
        }
    }

    /**
     * Handles adding a reply to an existing comment.
     * @param {string} parentCommentId - The ID of the parent comment.
     * @param {string} replyText - The text of the reply.
     * @param {HTMLElement} repliesContainer - The DOM element where replies should be appended.
     */
    async function handleAddReply(parentCommentId, replyText, repliesContainer) {
        if (!ADD_REPLY_URL) {
            console.log("Reply URL not configured!"); // Changed from alert()
            return;
        }
        try {
            const replyData = {
                parent_comment_id: parentCommentId,
                reply_text: replyText,
            };
            const response = await sendAjaxRequest(ADD_REPLY_URL, 'POST', replyData);
            if (response.success && response.reply) {
                const parentComment = commentsData.find(c => c.id === parentCommentId);
                if (parentComment) {
                    if (!parentComment.replies) {
                        parentComment.replies = [];
                    }
                    parentComment.replies.push(response.reply);
                    if (repliesContainer) {
                        const newReplyEl = createReplyElement(response.reply, parentCommentId); // Pass parentCommentId
                        if (newReplyEl) {
                            repliesContainer.appendChild(newReplyEl);
                            repliesContainer.scrollTop = repliesContainer.scrollHeight;
                        }
                    } else {
                        console.warn("Replies container not found for dynamically adding reply.");
                    }
                } else {
                    console.warn("Parent comment not found in local commentsData array after reply.");
                }
            } else {
                console.log('Failed to add reply: ' + (response.message || 'Unknown error')); // Changed from alert()
            }
        } catch (error) {
            console.log('Error adding reply: ' + error.message); // Changed from alert()
        }
    }

    async function handleUpdateCommentPosition(commentId, x_pos, y_pos) {
        const url = UPDATE_COMMENT_URL;
        try {
            const response = await sendAjaxRequest(url, 'POST', { comment_id: commentId, x_pos: x_pos, y_pos: y_pos });
            if (response.success) {
                // Update the local commentsData to reflect the new position
                const commentIndex = commentsData.findIndex(c => c.id === commentId);
                if (commentIndex !== -1) {
                    commentsData[commentIndex].x_pos = x_pos;
                    commentsData[commentIndex].y_pos = y_pos;
                }
                // console.log('Comment position updated successfully:', response.message); // Log success instead of alert
            } else {
                console.error('Failed to update comment position:', response.message || 'Unknown error'); // Log error instead of alert
                // No alert here to prevent popup from closing
            }
        } catch (error) {
            console.error('Error updating comment position:', error); // Log error instead of alert
            // No alert here to prevent popup from closing
        }
    }

    async function handleEditComment(commentId, newCommentText) {
        const url = EDIT_COMMENT_URL;
        try {
            const response = await sendAjaxRequest(url, 'POST', {
                comment_id: commentId,
                comment_text: newCommentText
            });
            if (response.success) {
                // Update the local commentsData
                const commentIndex = commentsData.findIndex(c => c.id === commentId);
                if (commentIndex !== -1) {
                    commentsData[commentIndex].Comment = newCommentText;
                }
                console.log('Comment updated successfully!'); // Changed from alert()
            } else {
                console.log('Failed to edit comment: ' + (response.message || 'Unknown error')); // Changed from alert()
            }
        } catch (error) {
            console.log('Error editing comment: ' + error.message); // Changed from alert()
        }
    }

    // Renamed for clarity, original `handleDeleteComment` now points to this for main comment deletion.
    async function showConfirmDeleteModal(targetType, idToDelete, parentId = null) {
        currentDeleteTarget = targetType; // 'comment' or 'reply'
        if (targetType === 'comment') {
            currentCommentIdToDelete = idToDelete;
            currentReplyIdToDelete = null; // Clear reply specific IDs
            currentParentCommentIdForReplyDeletion = null;
            deleteConfirmModal.querySelector('.modal-message').textContent = 'Are you sure you want to delete this comment and all its replies? This action cannot be undone.';
        } else if (targetType === 'reply') {
            currentReplyIdToDelete = idToDelete;
            currentParentCommentIdForReplyDeletion = parentId;
            currentCommentIdToDelete = null; // Clear comment specific ID
            deleteConfirmModal.querySelector('.modal-message').textContent = 'Are you sure you want to delete this reply? This action cannot be undone.';
        }

        if (deleteConfirmModal) {
            deleteConfirmModal.classList.remove('hidden', 'opacity-0');
            deleteConfirmModal.classList.add('opacity-100');
            const modalContent = deleteConfirmModal.querySelector('div:first-child');
            if (modalContent) {
                modalContent.classList.remove('scale-95');
                modalContent.classList.add('scale-100');
            }
        }
    }

    async function handleEditReply(replyId, parentCommentId, newReplyText) {
        if (!EDIT_REPLY_URL) {
            console.log("Edit Reply URL not configured!"); // Changed from alert()
            return;
        }
        try {
            const response = await sendAjaxRequest(EDIT_REPLY_URL, 'POST', {
                reply_id: replyId,
                parent_comment_id: parentCommentId,
                reply_text: newReplyText
            });
            if (response.success) {
                // Update the local commentsData
                const parentComment = commentsData.find(c => c.id === parentCommentId);
                if (parentComment && parentComment.replies) {
                    const replyIndex = parentComment.replies.findIndex(r => r.id === replyId); // Fixed typo from repllies to replies
                    if (replyIndex !== -1) {
                        parentComment.replies[replyIndex].Comment = newReplyText;
                    }
                }
                console.log('Reply updated successfully!'); // Changed from alert()
            } else {
                console.log('Failed to edit reply: ' + (response.message || 'Unknown error')); // Changed from alert()
            }
        } catch (error) {
            console.log('Error editing reply: ' + error.message); // Changed from alert()
        }
    }

    async function executeDeleteReply(replyId, parentCommentId) {
        if (!DELETE_REPLY_URL) {
            console.log("Delete Reply URL not configured!"); // Changed from alert()
            return;
        }

        try {
            const response = await sendAjaxRequest(DELETE_REPLY_URL, 'POST', {
                reply_id: replyId,
                parent_comment_id: parentCommentId
            });
            if (response.success) {
                // Remove the reply from the local data array
                const parentComment = commentsData.find(c => c.id === parentCommentId);
                if (parentComment && parentComment.replies) {
                    parentComment.replies = parentComment.replies.filter(reply => reply.id !== replyId);
                }
                // Remove the reply element from the DOM
                const replyEl = document.querySelector(`[data-reply-id="${replyId}"]`);
                if (replyEl) {
                    replyEl.remove();
                }

                // IMPORTANT: After deleting a reply, update the preview content's reply count for the parent comment
                const parentCommentIcon = document.getElementById(parentCommentId);
                if (parentCommentIcon) {
                    const previewContent = parentCommentIcon.querySelector('.absolute.top-full'); // Get the preview content div
                    if (previewContent) {
                        const updatedParentComment = commentsData.find(c => c.id === parentCommentId);
                        const updatedReplyCount = updatedParentComment.replies ? updatedParentComment.replies.length : 0;
                        const updatedReplyText = updatedReplyCount === 1 ? '1 reply' : `${updatedReplyCount} replies`;
                        previewContent.querySelector('p:last-child').textContent = updatedReplyText; // Update the reply count text
                    }
                }

                console.log('Reply deleted successfully!'); // Changed from alert()
                hideConfirmDeleteModal(); // Hide modal after successful deletion
            } else {
                console.log('Failed to delete reply: ' + (response.message || 'Unknown error')); // Changed from alert()
            }
        } catch (error) {
            console.log('Error deleting reply: ' + error.message); // Changed from alert()
        }
    }

    async function executeDeleteComment(commentId) {
        try {
            console.log(`Sending AJAX request to: ${DELETE_COMMENT_URL.replace('0', VIDEO_ID)}`);
            const response = await sendAjaxRequest(DELETE_COMMENT_URL.replace('0', VIDEO_ID), 'POST', {
                comment_id: commentId
            });

            console.log("AJAX response received:", response);

            if (response.status === 'success') {
                console.log("Comment deleted successfully. Closing popup and removing icon.");
                closeExpandedComment(); // Close the comment popup
                const commentIcon = document.getElementById(commentId);
                if (commentIcon) {
                    commentIcon.remove();
                }
                // Also remove the comment from the local commentsData array
                commentsData = commentsData.filter(c => c.id !== commentId);
                console.log('Comment and its replies deleted successfully!'); // Changed from alert()
                hideConfirmDeleteModal(); // Hide modal after successful deletion
            } else {
                console.error("Failed to delete comment:", response.message);
                console.log('Failed to delete comment: ' + (response.message || 'Unknown error')); // Changed from alert()
            }
        } catch (error) {
            console.error("Error in executeDeleteComment:", error);
            console.log('Error deleting comment: ' + error.message); // Changed from alert()
        }
    }


    // --- Event Listeners for the Add Comment Button ---
    if (addCommentBtn && commentAreaContainer) {
        addCommentBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent clicks on button from triggering document click handler

            // Disable commenting mode if already active or opening modal
            if (isCommentingMode) {
                isCommentingMode = false;
                addCommentBtn.classList.remove('bg-red-500', 'hover:bg-red-600');
                addCommentBtn.classList.add('bg-blue-500', 'hover:bg-blue-600');
                addCommentBtn.textContent = 'Add Comment';
                document.body.style.cursor = 'default';
                commentAreaContainer.removeEventListener('click', handleCommentAreaClick);
                hideCommentModal();
            } else {
                isCommentingMode = true;
                addCommentBtn.classList.remove('bg-blue-500', 'hover:bg-blue-600');
                addCommentBtn.classList.add('bg-red-500', 'hover:bg-red-600');
                addCommentBtn.textContent = 'Cancel Comment';
                document.body.style.cursor = 'crosshair'; // Change cursor to indicate drawing mode
                commentAreaContainer.addEventListener('click', handleCommentAreaClick);
            }
        });
    }

    function handleCommentAreaClick(e) {
        if (!isCommentingMode) return;

        // Check if the click occurred on an existing comment icon
        if (e.target.closest('.comment-icon') || e.target.closest('.absolute.z-30.w-8.h-8')) { // Added the actual class for comment icons
            return; // Do not add a new comment if clicking an existing one
        }

        // Calculate position relative to the comment area container
        const rect = draggableCommentsContainer.getBoundingClientRect();
        newCommentPlacement.x = e.clientX - rect.left;
        newCommentPlacement.y = e.clientY - rect.top;

        showCommentModal();
        isCommentingMode = false; // Exit commenting mode after placing a comment
        addCommentBtn.classList.remove('bg-red-500', 'hover:bg-red-600');
        addCommentBtn.classList.add('bg-blue-500', 'hover:bg-blue-600');
        addCommentBtn.textContent = 'Add Comment';
        document.body.style.cursor = 'default';
        commentAreaContainer.removeEventListener('click', handleCommentAreaClick);
    }


    if (saveCommentBtn) {
        saveCommentBtn.addEventListener('click', async () => {
            if (!commentInput) {
                return;
            }
            const commentText = commentInput.value.trim();
            if (commentText) { // This check should be redundant if button is disabled when empty
                await handleAddComment(commentText, newCommentPlacement.x, newCommentPlacement.y);
                hideCommentModal();
            } else {
                // This block should ideally not be hit if the button is disabled.
                // Keeping as console.log for defensive programming/debugging.
                const messageBox = document.createElement('div');
                messageBox.className = 'fixed inset-0 bg-red-600 bg-opacity-75 flex items-center justify-center text-white text-lg font-bold z-50 rounded-lg shadow-xl';
                messageBox.textContent = 'Comment text cannot be empty!';
                document.body.appendChild(messageBox);
                setTimeout(() => {
                    messageBox.remove();
                }, 2000);
                console.log("Comment text cannot be empty!");
            }
        });

        // Add input listener for the initial comment input
        commentInput.addEventListener('input', () => {
            toggleSaveButton(commentInput, saveCommentBtn);
        });
    }


    if (cancelCommentBtn) {
        cancelCommentBtn.addEventListener('click', () => {
            hideCommentModal();
        });
    }


    // --- Initial Rendering of Comments ---
    if (draggableCommentsContainer) {
        draggableCommentsContainer.innerHTML = '';
        commentsData.forEach(comment => {
            const commentEl = createCommentElement(comment);
            if (commentEl) {
                draggableCommentsContainer.appendChild(commentEl);
            }
        });
    }

    document.addEventListener('click', (e) => {
        if (currentOpenReplyDropdown && !currentOpenReplyDropdown.contains(e.target) && !e.target.closest('.reply-options-btn')) {
            currentOpenReplyDropdown.remove();
            currentOpenReplyDropdown = null;
        }
        // Global click listener to close main comment dropdown if clicked outside
        if (currentOpenCommentDropdown && !currentOpenCommentDropdown.contains(e.target) && !e.target.closest('.comment-options-btn')) {
            currentOpenCommentDropdown.remove();
            currentOpenCommentDropdown = null;
        }
        // Global click listener to close the comment popup if clicked outside
        // This is important because the popup is now appended to draggableCommentsContainer, not body.
        // And it no longer has a full-screen background to click on.
        // Added checks to prevent closing when clicking inside comment or reply option dropdowns.
        if (fullContentOverlay && !fullContentOverlay.contains(e.target) && !e.target.closest('.comment-icon') && !e.target.closest('.absolute.z-30.w-8.h-8') && !e.target.closest('.comment-options-dropdown') && !e.target.closest('.reply-options-dropdown')) {
            closeExpandedComment();
        }
    });

    // --- Delete Confirmation Modal Functionality ---
    // Add the HTML for the delete confirmation modal to the body
    const deleteConfirmModalHTML = `
        <div id="delete-confirm-modal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[300] hidden opacity-0 transition-opacity duration-300">
            <div class="bg-white p-6 rounded-lg shadow-xl max-w-sm w-full transform scale-95 transition-transform duration-300">
                <p class="text-lg font-semibold mb-4">Confirm Deletion</p>
                <p class="mb-6 modal-message">Are you sure you want to delete this comment and all its replies? This action cannot be undone.</p>
                <div class="flex justify-end space-x-3">
                    <button id="cancel-delete-btn" class="bg-gray-300 text-gray-800 px-4 py-2 rounded-md hover:bg-gray-400">Cancel</button>
                    <button id="confirm-delete-btn" class="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600">Delete</button>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', deleteConfirmModalHTML);

    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
    const cancelDeleteBtn = document.getElementById('cancel-delete-btn');
    const deleteConfirmModal = document.getElementById('delete-confirm-modal');
    const modalMessageElement = deleteConfirmModal.querySelector('.modal-message');

    function hideConfirmDeleteModal() {
        if (deleteConfirmModal) {
            deleteConfirmModal.classList.remove('opacity-100');
            deleteConfirmModal.classList.add('opacity-0');
            const modalContent = deleteConfirmModal.querySelector('div:first-child');
            if (modalContent) {
                modalContent.classList.remove('scale-100');
                modalContent.classList.add('scale-95');
            }
            setTimeout(() => {
                deleteConfirmModal.classList.add('hidden');
                currentCommentIdToDelete = null;
                currentReplyIdToDelete = null;
                currentParentCommentIdForReplyDeletion = null;
                currentDeleteTarget = null;
            }, 300); // Match CSS transition duration
        }
    }

    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', async (e) => { // Add 'e' parameter
            e.stopPropagation(); // Prevent click from bubbling up to document and closing popup
            if (currentDeleteTarget === 'comment' && currentCommentIdToDelete) {
                console.log(`Proceeding to delete comment with ID: ${currentCommentIdToDelete}`);
                await executeDeleteComment(currentCommentIdToDelete);
                // executeDeleteComment handles closing the popup and removing the icon itself
            } else if (currentDeleteTarget === 'reply' && currentReplyIdToDelete && currentParentCommentIdForReplyDeletion) {
                console.log(`Proceeding to delete reply with ID: ${currentReplyIdToDelete} from parent comment ${currentParentCommentIdForReplyDeletion}`);
                await executeDeleteReply(currentReplyIdToDelete, currentParentCommentIdForReplyDeletion);
                // executeDeleteReply does NOT close the popup, which is the desired behavior for replies
            }
        });
    }

    if (cancelDeleteBtn) {
        cancelDeleteBtn.addEventListener('click', (e) => { // Add 'e' parameter
            e.stopPropagation(); // Prevent click from bubbling up to document and closing popup
            hideConfirmDeleteModal();
            console.log("Delete action cancelled from modal.");
        });
    }

    // Close modal if clicking outside it (on the semi-transparent background)
    if (deleteConfirmModal) {
        deleteConfirmModal.addEventListener('click', (e) => {
            if (e.target === deleteConfirmModal) { // Check if click was directly on the overlay, not its children
                hideConfirmDeleteModal();
                console.log("Delete action cancelled by clicking outside modal.");
            }
        });
    }

}

document.addEventListener('DOMContentLoaded', initializeCommentsSystem);