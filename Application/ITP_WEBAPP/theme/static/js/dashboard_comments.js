// dashboard_comments.js

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

    const ADD_COMMENT_URL = commentDataElement.dataset.addCommentUrl;
    const UPDATE_COMMENT_URL = commentDataElement.dataset.updateCommentUrl;
    const DELETE_COMMENT_URL = commentDataElement.dataset.deleteCommentUrl;
    const ADD_REPLY_URL = commentDataElement.dataset.addReplyUrl;

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
     * Closes any currently expanded comment overlay.
     */
    function closeExpandedComment() {
        if (fullContentOverlay) {
            fullContentOverlay.classList.remove('opacity-100', 'scale-100', 'pointer-events-auto');
            fullContentOverlay.classList.add('opacity-0', 'scale-0', 'pointer-events-none', 'invisible');
            if (expandedCommentElement) {
                expandedCommentElement.classList.remove('is-expanded');
                makeDraggable(expandedCommentElement, expandedCommentElement.id, true);
                expandedCommentElement.style.cursor = 'grab'; // Reset cursor
                expandedCommentElement.style.zIndex = '30'; // Revert icon z-index
            }
            // Remove the overlay from DOM after transition completes (or immediately for quick feedback)
            setTimeout(() => { // Gives time for transition to play
                if (fullContentOverlay && fullContentOverlay.parentNode) {
                    fullContentOverlay.remove();
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
     * @returns {HTMLElement} The created reply element.
     */
    function createReplyElement(reply) {
        const replyDiv = document.createElement('div');
        replyDiv.className = 'flex items-start mb-2 last:mb-0 ml-4 border-l pl-3 border-gray-200 py-1'; // Indent replies
        replyDiv.innerHTML = `
            <div class="w-8 h-8 bg-gray-200 text-gray-700 rounded-full flex items-center justify-center text-xs font-bold mr-2 flex-shrink-0">
                ${reply.CommentedBy ? reply.CommentedBy.charAt(0).toUpperCase() : 'U'}
            </div>
            <div>
                <p class="font-semibold text-sm">${reply.CommentedBy || 'Unknown User'} <span class="text-xs text-gray-500 font-normal ml-1">${reply.FormattedDate || 'No Date'}</span></p>
                <p class="text-sm text-gray-700">${reply.Comment}</p>
            </div>
        `;
        return replyDiv;
    }

    /**
     * Creates and appends a draggable comment icon with hover preview and click-to-expand functionality.
     * @param {object} comment - The comment object {id, Comment, CommentedBy, FormattedDate, x_pos, y_pos, replies}.
     */
    function createCommentElement(comment) {
        // Ensure x_pos and y_pos are numbers. If null, don't create visual icon.
        // For comments intended for table, x_pos/y_pos will be null, so this function won't create a visual marker.
        // Instead, the table logic will need to handle rendering those.
        const x_pos = parseFloat(comment.x_pos);
        const y_pos = parseFloat(comment.y_pos);

        if (isNaN(x_pos) || isNaN(y_pos)) {
            // This comment is likely a non-positional comment (e.g., from a table)
            // It should not have a visual icon on the video.
            // You'll need separate logic to render these in your table view.
            return null; // Return null if comment is not position-based
        }

        // 1. The main comment container (now the visual icon with solid black border)
        const commentContainerDiv = document.createElement('div');
        commentContainerDiv.id = comment.id;
        commentContainerDiv.className = 'absolute z-30 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold border-2 border-black hover:bg-blue-600 transition-colors duration-200 focus:outline-none focus:ring-0 cursor-grab';
        commentContainerDiv.style.left = `${x_pos}px`;
        commentContainerDiv.style.top = `${y_pos}px`;
        commentContainerDiv.style.transform = 'translate(-50%, -50%)'; // Center the icon on the point

        // The SVG element directly inside the commentContainerDiv
        commentContainerDiv.innerHTML = `<svg class="w-6 h-6" fill="white" stroke="black" stroke-width="1.5" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.336-3.11c-.813-1.013-1.336-2.31-1.336-3.89C2 6.134 5.582 3 10 3s8 3.134 8 7z" clip-rule="evenodd"></path></svg>`;

        // 2. Preview content (hidden by default, shown on hover)
        const previewContent = document.createElement('div');
        previewContent.className = 'absolute top-full left-1/2 -translate-x-1/2 mt-2 p-2 bg-white rounded-lg shadow-lg text-xs text-gray-800 opacity-0 scale-0 origin-top transition-all duration-200 ease-out pointer-events-none w-48 max-w-xs overflow-hidden invisible';
        previewContent.innerHTML = `
            <p class="font-semibold mb-1">${comment.CommentedBy || 'Unknown User'}</p>
            <p style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${truncateText(comment.Comment, 50)}</p>
        `;
        commentContainerDiv.appendChild(previewContent); // Append preview to the container

        // 3. Full content overlay (hidden by default, shown on click)
        const fullContentOverlayElement = document.createElement('div');
        fullContentOverlayElement.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[100] opacity-0 scale-0 transition-all duration-300 ease-out pointer-events-none invisible';

        const fullContentCard = document.createElement('div');
        fullContentCard.className = 'bg-white rounded-lg shadow-2xl p-6 w-96 max-w-sm transform transition-all duration-300 ease-out flex flex-col max-h-[90vh]'; // Added flex flex-col and max-h for scroll
        
        // --- Inner HTML for the full comment card ---
        fullContentCard.innerHTML = `
            <div class="flex items-start justify-between mb-3 flex-shrink-0">
                <div class="flex items-center">
                    <div class="w-10 h-10 bg-gray-300 text-gray-800 rounded-full flex items-center justify-center text-md font-bold mr-3">
                        ${comment.CommentedBy ? comment.CommentedBy.charAt(0).toUpperCase() : 'U'}
                    </div>
                    <div>
                        <p class="font-semibold text-base">${comment.CommentedBy || 'Unknown User'}</p>
                        <p class="text-xs text-gray-500">${comment.FormattedDate || 'No Date'}</p>
                    </div>
                </div>
                <button class="close-comment-btn p-1 rounded-full hover:bg-gray-100 focus:outline-none">
                    <svg class="w-6 h-6 text-gray-500 hover:text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                </button>
            </div>
            <p class="text-sm mb-4 flex-shrink-0">${comment.Comment}</p>

            <div class="replies-container overflow-y-auto flex-grow pr-2 mb-4">
                </div>

            <div class="flex items-center mb-3 flex-shrink-0">
                <input type="text" placeholder="Add a reply..." class="reply-input flex-grow p-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                <button class="reply-btn ml-2 bg-blue-500 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-600 transition-colors duration-200">Reply</button>
            </div>

            <button class="delete-comment-btn w-full bg-red-500 text-white py-2 rounded-md text-sm hover:bg-red-600 transition-colors duration-200 mt-2 flex-shrink-0">Delete Comment</button>
        `;
        fullContentOverlayElement.appendChild(fullContentCard);

        // Populate replies if they exist
        const repliesContainer = fullContentCard.querySelector('.replies-container');
        if (comment.replies && Array.isArray(comment.replies) && repliesContainer) {
            comment.replies.forEach(reply => {
                const replyEl = createReplyElement(reply);
                if (replyEl) { // Ensure replyEl is not null
                    repliesContainer.appendChild(replyEl);
                }
            });
        }


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
            if (e.target.closest('.full-content-overlay-element') ||
                e.target.closest('.delete-comment-btn') ||
                e.target.closest('.close-comment-btn') ||
                e.target.closest('.reply-btn') ||
                e.target.closest('.reply-input')) {
                return;
            }

            if (!commentContainerDiv.classList.contains('is-expanded')) {
                closeExpandedComment(); // Close any other open comment

                expandedCommentElement = commentContainerDiv;
                fullContentOverlay = fullContentOverlayElement; // Assign the new overlay element

                document.body.appendChild(fullContentOverlay);
                fullContentOverlay.classList.remove('opacity-0', 'scale-0', 'pointer-events-none', 'invisible');
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

        fullContentOverlayElement.querySelector('.delete-comment-btn').addEventListener('click', (event) => {
            event.stopPropagation();
            handleDeleteComment(comment.id);
        });

        fullContentOverlayElement.querySelector('.reply-btn').addEventListener('click', async () => {
            const replyInput = fullContentOverlayElement.querySelector('.reply-input');
            const replyText = replyInput.value.trim();
            if (replyText) {
                // Pass the repliesContainer element to handleAddReply
                await handleAddReply(comment.id, replyText, repliesContainer);
                replyInput.value = ''; // Clear input after sending
                // No longer closeExpandedComment here, as we want to see the new reply
            } else {
                alert("Reply cannot be empty!");
            }
        });

        fullContentOverlayElement.addEventListener('click', (e) => {
            if (e.target === fullContentOverlayElement) { // Only close if clicking on the dim background
                closeExpandedComment();
            }
        });

        // Ensure draggableCommentsContainer is not null before appending
        if (draggableCommentsContainer) {
            draggableCommentsContainer.appendChild(commentContainerDiv);
            makeDraggable(commentContainerDiv, comment.id, true); // Enable dragging initially
        }

        return commentContainerDiv; // Return the created comment icon element
    }

    /**
     * Makes an element draggable within its parent container.
     * @param {HTMLElement} element - The element to make draggable.
     * @param {string} commentId - The ID of the comment associated with this element.
     * @param {boolean} enable - True to enable dragging, false to disable.
     */
    function makeDraggable(element, commentId, enable) {
        // Clean up previous listeners if they exist to prevent duplicates
        if (element._dragMouseDownHandler) {
            element.removeEventListener('mousedown', element._dragMouseDownHandler);
            document.removeEventListener('mousemove', element._dragMouseMoveHandler);
            document.removeEventListener('mouseup', element._dragMouseUpHandler);
            delete element._dragMouseDownHandler;
            delete element._dragMouseMoveHandler;
            delete element._dragMouseUpHandler;
        }

        if (enable) {
            let isDraggingElement = false;
            let offsetX, offsetY;

            element._dragMouseDownHandler = (e) => {
                // Only start drag if left mouse button, not on a button, and not expanded
                if (e.button === 0 && !e.target.closest('button') && !element.classList.contains('is-expanded')) {
                    isDraggingElement = true;
                    element.style.cursor = 'grabbing';
                    const elementRect = element.getBoundingClientRect();
                    offsetX = e.clientX - elementRect.left;
                    offsetY = e.clientY - elementRect.top;

                    // Remove current transform to calculate position directly from left/top
                    element.style.transform = 'none';

                    // Set initial left/top relative to its parent (draggableCommentsContainer)
                    const parentContainerRect = draggableCommentsContainer.getBoundingClientRect();
                    element.style.left = `${elementRect.left - parentContainerRect.left}px`;
                    element.style.top = `${elementRect.top - parentContainerRect.top}px`;

                    e.stopPropagation(); // Prevent document click from triggering
                }
            };

            element._dragMouseMoveHandler = (e) => {
                if (!isDraggingElement) return;

                if (!draggableCommentsContainer) {
                    console.error("draggableCommentsContainer is null during drag.");
                    return;
                }
                const parentContainerRect = draggableCommentsContainer.getBoundingClientRect();

                let newX = e.clientX - parentContainerRect.left - offsetX;
                let newY = e.clientY - parentContainerRect.top - offsetY;

                // Keep element within parent bounds
                const elementWidth = element.offsetWidth;
                const elementHeight = element.offsetHeight;

                newX = Math.max(0, newX);
                newX = Math.min(newX, parentContainerRect.width - elementWidth);

                newY = Math.max(0, newY);
                newY = Math.min(newY, parentContainerRect.height - elementHeight);

                element.style.left = `${newX}px`;
                element.style.top = `${newY}px`;

                // Update commentsData directly as well for persistence
                const commentIndex = commentsData.findIndex(c => c.id === commentId);
                if (commentIndex !== -1) {
                    commentsData[commentIndex].x_pos = newX;
                    commentsData[commentIndex].y_pos = newY;
                }
            };

            element._dragMouseUpHandler = () => {
                if (isDraggingElement) {
                    isDraggingElement = false;
                    element.style.cursor = 'grab'; // Reset cursor

                    // The element's position is already in pixels relative to its parent
                    const finalX = parseFloat(element.style.left);
                    const finalY = parseFloat(element.style.top);

                    // Re-apply the transform for centering visual, but update DB with raw pixel values
                    // If you want to store normalized values (0-1), you'd convert here.
                    // If your backend expects raw pixels, this is fine.
                    element.style.transform = 'translate(-50%, -50%)'; // Re-center the icon

                    handleUpdateCommentPosition(commentId, finalX, finalY);
                }
            };

            element.addEventListener('mousedown', element._dragMouseDownHandler);
            document.addEventListener('mousemove', element._dragMouseMoveHandler);
            document.addEventListener('mouseup', element._dragMouseUpHandler);
        } else {
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
                const newCommentEl = createCommentElement(response.comment);
                if (newCommentEl) { // Only append if it's a positional comment
                    // No need to append newCommentEl here, createCommentElement already does it
                    // if it's a position-based comment, otherwise it returns null.
                    // Initial render loop handles appending.
                }
            }
        } catch (error) {
            alert('Failed to add comment: ' + error.message);
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
            alert("Reply URL not configured!");
            return;
        }
        try {
            const replyData = {
                parent_comment_id: parentCommentId,
                reply_text: replyText,
            };

            const response = await sendAjaxRequest(ADD_REPLY_URL, 'POST', replyData);

            if (response.success && response.reply) {
                // Find the parent comment in the local commentsData array and add the reply
                const parentComment = commentsData.find(c => c.id === parentCommentId);
                if (parentComment) {
                    if (!parentComment.replies) {
                        parentComment.replies = [];
                    }
                    parentComment.replies.push(response.reply);

                    // Dynamically add the new reply to the DOM
                    if (repliesContainer) {
                        const newReplyEl = createReplyElement(response.reply);
                        if (newReplyEl) {
                            repliesContainer.appendChild(newReplyEl);
                            repliesContainer.scrollTop = repliesContainer.scrollHeight; // Scroll to bottom
                        }
                    } else {
                        console.warn("Replies container not found for dynamically adding reply.");
                    }
                } else {
                    console.warn("Parent comment not found in local commentsData array after reply.");
                }
            } else {
                alert('Failed to add reply: ' + (response.message || 'Unknown error'));
            }
        } catch (error) {
            alert('Error adding reply: ' + error.message);
        }
    }

    async function handleUpdateCommentPosition(commentId, x_pos, y_pos) {
        const url = UPDATE_COMMENT_URL;
        try {
            const response = await sendAjaxRequest(url, 'POST', { comment_id: commentId, x_pos: x_pos, y_pos: y_pos });
            // console.log("Update position response:", response); // For debugging
        } catch (error) {
            alert('Failed to update comment position: ' + error.message);
        }
    }

    async function handleDeleteComment(commentId) {
        if (!confirm('Are you sure you want to delete this comment?')) {
            return;
        }
        const url = DELETE_COMMENT_URL;
        try {
            const response = await sendAjaxRequest(url, 'POST', { comment_id: commentId });
            const commentElement = document.getElementById(commentId);
            if (commentElement) {
                commentElement.remove();
            }
            commentsData = commentsData.filter(comment => comment.id !== commentId);
            closeExpandedComment();
        } catch (error) {
            alert('Failed to delete comment: ' + error.message);
        }
    }

    // --- Event Listeners ---

    // This listener on the container handles clicks for new comment placement or closing.
    if (draggableCommentsContainer) {
        draggableCommentsContainer.addEventListener('click', (e) => {
            if (isCommentingMode) {
                // If clicking directly on the draggableCommentsContainer (the video overlay)
                if (e.target === draggableCommentsContainer) {
                    const containerRect = draggableCommentsContainer.getBoundingClientRect(); // Use draggableCommentsContainer for coords
                    newCommentPlacement.x = e.clientX - containerRect.left;
                    newCommentPlacement.y = e.clientY - containerRect.top;

                    showCommentModal();
                    isCommentingMode = false;
                    if (addCommentBtn) {
                        addCommentBtn.textContent = 'Add Comment';
                        addCommentBtn.classList.remove('bg-blue-600', 'hover:bg-blue-500', 'border-blue-700');
                        addCommentBtn.classList.add('bg-gray-500', 'hover:bg-gray-400', 'border-gray-600');
                    }
                    if (commentAreaContainer) { // Assuming commentAreaContainer is the parent of draggableCommentsContainer
                        commentAreaContainer.style.cursor = 'default';
                    }
                    if (draggableCommentsContainer) {
                        draggableCommentsContainer.classList.remove('bg-black', 'bg-opacity-10');
                    }
                }
            } else {
                // If not in commenting mode, and clicking outside an expanded comment, close any open comment
                if (e.target === draggableCommentsContainer && fullContentOverlay) {
                    closeExpandedComment();
                }
            }
        });
    }


    // Ensure addCommentBtn is not null before adding listener
    if (addCommentBtn) {
        addCommentBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent this click from propagating to draggableCommentsContainer
            isCommentingMode = !isCommentingMode;

            if (isCommentingMode) {
                closeExpandedComment(); // Close any expanded comment when entering commenting mode
                addCommentBtn.textContent = 'Click to Place Comment (Active)';
                addCommentBtn.classList.remove('bg-gray-500', 'hover:bg-gray-400', 'border-gray-600');
                addCommentBtn.classList.add('bg-blue-600', 'hover:bg-blue-500', 'border-blue-700');
                if (commentAreaContainer) {
                    commentAreaContainer.style.cursor = 'crosshair';
                }
                if (draggableCommentsContainer) {
                    draggableCommentsContainer.classList.add('bg-black', 'bg-opacity-10');
                }
            } else {
                addCommentBtn.textContent = 'Add Comment';
                addCommentBtn.classList.remove('bg-blue-600', 'hover:bg-blue-500', 'border-blue-700');
                addCommentBtn.classList.add('bg-gray-500', 'hover:bg-gray-400', 'border-gray-600');
                if (commentAreaContainer) {
                    commentAreaContainer.style.cursor = 'default';
                }
                if (draggableCommentsContainer) {
                    draggableCommentsContainer.classList.remove('bg-black', 'bg-opacity-10');
                }
            }
        });
    }


    // Ensure saveCommentBtn is not null before adding listener
    if (saveCommentBtn) {
        saveCommentBtn.addEventListener('click', async () => {
            if (!commentInput) {
                return;
            }
            const commentText = commentInput.value.trim();
            if (commentText) {
                await handleAddComment(commentText, newCommentPlacement.x, newCommentPlacement.y);
                hideCommentModal();
            } else {
                const messageBox = document.createElement('div');
                messageBox.className = 'fixed inset-0 bg-red-600 bg-opacity-75 flex items-center justify-center text-white text-lg font-bold z-50 rounded-lg shadow-xl';
                messageBox.textContent = 'Comment text cannot be empty!';
                document.body.appendChild(messageBox);
                setTimeout(() => {
                    messageBox.remove();
                }, 2000);
            }
        });
    }


    // Ensure cancelCommentBtn is not null before adding listener
    if (cancelCommentBtn) {
        cancelCommentBtn.addEventListener('click', () => {
            hideCommentModal();
        });
    }


    // --- Initial Rendering of Comments ---
    // Clear existing comments and render all from commentsData
    if (draggableCommentsContainer) {
        draggableCommentsContainer.innerHTML = '';
        commentsData.forEach(comment => {
            const commentEl = createCommentElement(comment);
            // Only append to draggableCommentsContainer if it's a positional comment (not null)
            if (commentEl) {
                draggableCommentsContainer.appendChild(commentEl);
            }
        });
    }

}

document.addEventListener('DOMContentLoaded', initializeCommentsSystem);