// dashboard_results.js

document.addEventListener('DOMContentLoaded', () => {
    // Select both video players
    const vid1 = document.getElementById('video-player'); // Left video (master)
    const vid2 = document.getElementById('video-player-right'); // Right video (slave)

    // Only proceed if *both* video players exist on the page
    if (vid1 && vid2) {
        let seeking = false;

        function syncTime(sourceVideo, targetVideo) {
            if (!seeking) {
                seeking = true;
                targetVideo.currentTime = sourceVideo.currentTime;
                // Use a short timeout to prevent event loops
                setTimeout(() => seeking = false, 100);
            }
        }

        // --- Sync Time (Seeking) ---
        // Add listeners for both seeking and seeked events for robustness
        ['seeking', 'seeked'].forEach(evt => {
            vid1.addEventListener(evt, () => syncTime(vid1, vid2));
            vid2.addEventListener(evt, () => syncTime(vid2, vid1));
        });

        // --- [FIX] Sync Time (Continuous Playback) ---
        // Use 'timeupdate' on the master video (vid1) to continuously correct for playback drift.
        // We listen to vid1 (master) and correct vid2 (slave).
        vid1.addEventListener('timeupdate', () => {
            // Check if both videos are currently playing and no manual seek is in progress
            if (!vid1.paused && !vid2.paused && !seeking) {
                const timeDiff = Math.abs(vid1.currentTime - vid2.currentTime);
                // Only correct if the difference exceeds a small threshold (e.g., 0.1s)
                // to prevent excessive seeking/stuttering.
                if (timeDiff > 0.1) { 
                    // Correct vid2 to vid1's time
                    vid2.currentTime = vid1.currentTime;
                }
            }
        });
        
        // --- Sync State (Play/Pause) ---
        vid1.addEventListener('play', () => {
            if (vid2.paused) vid2.play();
        });
        vid1.addEventListener('pause', () => {
            if (!vid2.paused) vid2.pause();
        });

        vid2.addEventListener('play', () => {
            if (vid1.paused) vid1.play();
        });
        vid2.addEventListener('pause', () => {
            if (!vid1.paused) vid1.pause();
        });
    }
});