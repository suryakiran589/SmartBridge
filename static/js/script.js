document.addEventListener('DOMContentLoaded', () => {
    const gestureText = document.getElementById('gesture-text');
    const actionText = document.getElementById('action-text');
    const confidenceText = document.getElementById('confidence-text');
    const confidenceBar = document.getElementById('confidence-bar');
    const statCards = document.querySelectorAll('.stat-card');

    // Function to update the UI based on data from Flask API
    const updateDashboard = (data) => {
        // Update Text
        gestureText.textContent = data.gesture;
        actionText.textContent = data.action;

        // Update Confidence
        const confValue = parseFloat(data.confidence) * 100;
        confidenceText.textContent = `${confValue.toFixed(0)}%`;
        confidenceBar.style.width = `${confValue}%`;

        // Add visual flair based on state
        if (data.gesture !== "None" && data.gesture !== "Waiting...") {
            statCards[0].style.borderColor = "rgba(96, 165, 250, 0.5)"; // Blue highlight
            statCards[0].style.boxShadow = "0 0 15px rgba(96, 165, 250, 0.2)";
        } else {
            statCards[0].style.borderColor = "rgba(255, 255, 255, 0.1)"; // Reset
            statCards[0].style.boxShadow = "none";
        }

        if (data.action !== "Waiting..." && data.action !== "None") {
            statCards[1].style.borderColor = "rgba(167, 139, 250, 0.5)"; // Purple highlight
            statCards[1].style.boxShadow = "0 0 15px rgba(167, 139, 250, 0.2)";
        } else {
            statCards[1].style.borderColor = "rgba(255, 255, 255, 0.1)"; // Reset
            statCards[1].style.boxShadow = "none";
        }
    };

    // Polling function
    const fetchCurrentState = async () => {
        try {
            const response = await fetch('/api/current_state');
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            updateDashboard(data);
        } catch (error) {
            console.error('Error fetching state:', error);
            // Optionally update UI to indicate connection loss
            gestureText.textContent = "Connection Lost";
            actionText.textContent = "Check Server";
        }
    };

    // Poll every 200 milliseconds (5 times a second)
    setInterval(fetchCurrentState, 200);
});
