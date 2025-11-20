document.addEventListener("DOMContentLoaded", function () {
    const videoPlayer = document.getElementById("video-player");
    // [FIX] Get the second video player as well for simultaneous seeking
    const videoPlayerRight = document.getElementById("video-player-right"); 
    const rows = document.querySelectorAll("tbody tr");

    rows.forEach(row => {
        row.addEventListener("click", function () {
            const timestamp = parseFloat(row.dataset.timestamp); // Get timestamp from data attribute
            if (!isNaN(timestamp)) {
                if (videoPlayer) {
                    // Seek the main video
                    videoPlayer.currentTime = timestamp; 
                }
                if (videoPlayerRight) {
                    // [FIX] Directly seek the second video, ensuring both start seeking immediately
                    videoPlayerRight.currentTime = timestamp;
                }
            }
        });
    });
});