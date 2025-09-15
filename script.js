document.addEventListener('DOMContentLoaded', function() {
    // Matrix Rain Effect
    const canvas = document.getElementById('matrix-rain');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to window size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Characters for matrix rain - more tech-looking characters
    const characters = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789<>[]{}|/\\=+-*&^%$#@!';
    
    // Font size and columns
    const fontSize = 14;
    const columns = Math.floor(canvas.width / fontSize);
    
    // Array to track the y position of each drop
    const drops = [];
    for (let i = 0; i < columns; i++) {
        drops[i] = Math.random() * -100;
    }
    
    // Draw the matrix rain
    function drawMatrixRain() {
        // Semi-transparent black to create trail effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Set text color and font
        ctx.fillStyle = '#0f0';
        ctx.font = fontSize + 'px "Courier New", monospace';
        
        // Draw characters
        for (let i = 0; i < drops.length; i++) {
            // Random character
            const text = characters.charAt(Math.floor(Math.random() * characters.length));
            
            // x coordinate of the drop
            const x = i * fontSize;
            // y coordinate of the drop
            const y = drops[i] * fontSize;
            
            // Randomly change color for some characters
            if (Math.random() > 0.975) {
                ctx.fillStyle = '#00ff9d'; // Light green
            } else if (Math.random() > 0.95) {
                ctx.fillStyle = '#00b36b'; // Medium green
            } else {
                ctx.fillStyle = '#0f0'; // Bright green
            }
            
            // Vary the brightness for a more dynamic effect
            const brightness = Math.random() * 0.5 + 0.5;
            ctx.globalAlpha = brightness;
            
            // Draw the character
            ctx.fillText(text, x, y);
            
            // Reset globalAlpha
            ctx.globalAlpha = 1.0;
            
            // Reset drop to top if it reaches bottom or randomly
            if (y > canvas.height || Math.random() > 0.975) {
                drops[i] = 0;
            }
            
            // Move drop down
            drops[i]++;
        }
    }
    
    // Animation loop
    function animate() {
        drawMatrixRain();
        requestAnimationFrame(animate);
    }
    
    // Start animation
    animate();
    
    // Resize canvas when window is resized
    window.addEventListener('resize', function() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        // Recalculate columns
        const newColumns = Math.floor(canvas.width / fontSize);
        
        // Adjust drops array
        if (newColumns > drops.length) {
            // Add new drops
            for (let i = drops.length; i < newColumns; i++) {
                drops[i] = Math.random() * -100;
            }
        } else {
            // Remove extra drops
            drops.length = newColumns;
        }
    });
    
    // Try button click event
    const tryButton = document.querySelector('.try-button');
    tryButton.addEventListener('click', function() {
        alert('Guardian Eye software will launch here!');
    });
    
    // Eye logo hover effect and realistic eye movement
    const eyeLogo = document.getElementById('eye-logo');
    const featureDetails = document.querySelector('.feature-details');
    const eyePupil = document.querySelector('.eye-pupil');
    const eyeInner = document.querySelector('.eye-inner');
    
    // Eye hover effect for feature details
    eyeLogo.addEventListener('mouseenter', function() {
        featureDetails.style.display = 'block';
        setTimeout(() => {
            featureDetails.style.opacity = '1';
        }, 10);
    });
    
    eyeLogo.addEventListener('mouseleave', function() {
        featureDetails.style.opacity = '0';
        setTimeout(() => {
            featureDetails.style.display = 'none';
        }, 500);
    });
    
    // Realistic eye tracking movement
    document.addEventListener('mousemove', function(e) {
        const eyeRect = eyeLogo.getBoundingClientRect();
        const eyeCenterX = eyeRect.left + eyeRect.width / 2;
        const eyeCenterY = eyeRect.top + eyeRect.height / 2;
        
        // Calculate distance from eye center to cursor
        const dx = e.clientX - eyeCenterX;
        const dy = e.clientY - eyeCenterY;
        
        // Calculate angle and distance
        const angle = Math.atan2(dy, dx);
        const distance = Math.min(Math.sqrt(dx*dx + dy*dy) / 100, 1);
        
        // Calculate pupil movement (limited to stay within the eye)
        const maxMove = 10; // maximum pixels to move
        const moveX = Math.cos(angle) * distance * maxMove;
        const moveY = Math.sin(angle) * distance * maxMove;
        
        // Apply movement to pupil and iris
        eyePupil.style.transform = `translate(${moveX}px, ${moveY}px)`;
        eyeInner.style.transform = `translate(${moveX * 0.5}px, ${moveY * 0.5}px)`;
    });
    
    // Occasional random eye movement for realism
    function randomEyeMovement() {
        if (!document.hasFocus()) return; // Don't move if page not focused
        
        const randomX = (Math.random() - 0.5) * 10;
        const randomY = (Math.random() - 0.5) * 10;
        
        eyePupil.style.transform = `translate(${randomX}px, ${randomY}px)`;
        eyeInner.style.transform = `translate(${randomX * 0.5}px, ${randomY * 0.5}px)`;
        
        // Reset after a short time
        setTimeout(() => {
            eyePupil.style.transform = 'translate(0, 0)';
            eyeInner.style.transform = 'translate(0, 0)';
        }, 300);
    }
    
    // Trigger random movement occasionally
    setInterval(randomEyeMovement, 5000);
});