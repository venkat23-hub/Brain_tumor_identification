// Brain Tumor Detection - Client-side JavaScript
// This file handles any additional client-side interactions

document.addEventListener('DOMContentLoaded', function() {
    console.log('Brain Tumor Detection App Loaded');
    
    // Smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Image preview enhancement
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    console.log('File loaded:', file.name);
                    // Add any custom preview logic here
                };
                reader.readAsDataURL(file);
            }
        });
    }
});

// Accessibility improvements
function improveAccessibility() {
    // Add ARIA labels where needed
    const cards = document.querySelectorAll('.result-card');
    cards.forEach((card, index) => {
        if (!card.getAttribute('role')) {
            card.setAttribute('role', 'region');
            card.setAttribute('aria-label', `Result ${index + 1}`);
        }
    });
}

// Initialize accessibility on load
window.addEventListener('load', improveAccessibility);
