/**
 * ============================================
 * RetainIQ - Main JavaScript
 * Professional & Clean Functionality
 * ============================================
 */

// ---------- IIFE - Immediately Invoked Function Expression ----------
(function() {
    'use strict';

    // ---------- DOM Ready ----------
    document.addEventListener('DOMContentLoaded', function() {
        console.log('RetainIQ Application Initialized');
        
        // Initialize all components
        initDarkMode();
        initScrollTop();
        initCounterAnimation();
        initFormValidation();
        initLoadingSpinner();
        initToastNotifications();
        initSmoothScroll();
        initNavbarScroll();
        initFAQAccordion();
        initContactForm();
        initScrollProgress();
        initTooltips();
        initFormAutoSave();
        initPrintButton();
    });

    // ---------- Dark Mode Toggle ----------
    function initDarkMode() {
        const toggle = document.getElementById('darkModeToggle');
        if (!toggle) return;

        // Check saved preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.documentElement.setAttribute('data-bs-theme', 'dark');
            toggle.innerHTML = '<i class="fas fa-sun"></i>';
            toggle.classList.add('active');
        }

        toggle.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-bs-theme');
            if (currentTheme === 'dark') {
                document.documentElement.setAttribute('data-bs-theme', 'light');
                localStorage.setItem('theme', 'light');
                this.innerHTML = '<i class="fas fa-moon"></i>';
                this.classList.remove('active');
                showToast('Light mode activated', 'info');
            } else {
                document.documentElement.setAttribute('data-bs-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                this.innerHTML = '<i class="fas fa-sun"></i>';
                this.classList.add('active');
                showToast('Dark mode activated', 'info');
            }
        });
    }

    // ---------- Scroll to Top Button ----------
    function initScrollTop() {
        const btn = document.getElementById('scrollTopBtn');
        if (!btn) return;

        window.addEventListener('scroll', function() {
            if (window.scrollY > 300) {
                btn.classList.add('visible');
            } else {
                btn.classList.remove('visible');
            }
        });

        btn.addEventListener('click', function() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // ---------- Counter Animation ----------
    function initCounterAnimation() {
        const counters = document.querySelectorAll('.counter');
        if (!counters.length) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const target = parseInt(entry.target.getAttribute('data-target'));
                    animateCounter(entry.target, target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        counters.forEach(counter => observer.observe(counter));
    }

    function animateCounter(element, target) {
        let current = 0;
        const increment = target / 50;
        const duration = 2000;
        const stepTime = duration / 50;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target + '%';
                clearInterval(timer);
            } else {
                element.textContent = Math.round(current) + '%';
            }
        }, stepTime);
    }

    // ---------- Form Validation ----------
    function initFormValidation() {
        const form = document.getElementById('predictionForm');
        if (!form) return;

        // Add real-time validation
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
            
            input.addEventListener('input', function() {
                if (this.classList.contains('is-invalid')) {
                    validateField(this);
                }
            });
        });

        form.addEventListener('submit', function(e) {
            // Show loading
            showLoading();

            // Validate all fields
            let isValid = true;
            const requiredFields = form.querySelectorAll('[required]');
            
            requiredFields.forEach(field => {
                if (!validateField(field)) {
                    isValid = false;
                }
            });

            if (!isValid) {
                e.preventDefault();
                hideLoading();
                showToast('Please fill in all required fields correctly.', 'error');
                return;
            }

            // Validate numeric fields
            const numberFields = form.querySelectorAll('input[type="number"]');
            let numericValid = true;
            numberFields.forEach(field => {
                const value = parseFloat(field.value);
                if (isNaN(value) || value < 0) {
                    e.preventDefault();
                    hideLoading();
                    field.classList.add('is-invalid');
                    showToast('Please enter valid numbers (positive values only).', 'error');
                    numericValid = false;
                    return;
                }
            });

            if (!numericValid) {
                return;
            }

            // Show loading overlay
            showLoading();
        });
    }

    function validateField(field) {
        if (!field.hasAttribute('required')) return true;
        
        const value = field.value.trim();
        if (!value) {
            field.classList.add('is-invalid');
            return false;
        }
        
        field.classList.remove('is-invalid');
        return true;
    }

    // ---------- Loading Spinner ----------
    function initLoadingSpinner() {
        // Hide loading overlay on page load
        hideLoading();
        
        // Hide on page load complete
        window.addEventListener('load', function() {
            hideLoading();
        });
    }

    function showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('active');
            overlay.style.display = 'flex';
        }
    }

    function hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
            overlay.style.display = 'none';
        }
    }

    // ---------- Toast Notifications ----------
    function initToastNotifications() {
        // Create toast container if it doesn't exist
        if (!document.querySelector('.toast-container')) {
            const container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
    }

    function showToast(message, type = 'info') {
        const container = document.querySelector('.toast-container');
        if (!container) return;

        const icons = {
            'success': 'fa-check-circle',
            'error': 'fa-exclamation-circle',
            'warning': 'fa-exclamation-triangle',
            'info': 'fa-info-circle'
        };

        const toast = document.createElement('div');
        toast.className = `toast-custom ${type}`;
        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.info}"></i>
            <span>${message}</span>
            <button type="button" class="btn-close btn-sm" onclick="this.parentElement.remove()"></button>
        `;

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100px)';
                setTimeout(() => toast.remove(), 300);
            }
        }, 5000);
    }

    // ---------- Smooth Scroll ----------
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const target = document.querySelector(targetId);
                if (target) {
                    e.preventDefault();
                    const navHeight = document.querySelector('.navbar')?.offsetHeight || 70;
                    const targetPosition = target.getBoundingClientRect().top + window.scrollY - navHeight;
                    
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    // ---------- Navbar Scroll Effect ----------
    function initNavbarScroll() {
        const navbar = document.querySelector('.navbar-glass');
        if (!navbar) return;

        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.style.boxShadow = '0 4px 30px rgba(0, 0, 0, 0.1)';
                navbar.style.padding = '0.5rem 0';
            } else {
                navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.05)';
                navbar.style.padding = '1rem 0';
            }
        });
    }

    // ---------- FAQ Accordion ----------
    function initFAQAccordion() {
        document.querySelectorAll('.accordion-button').forEach(button => {
            button.addEventListener('click', function() {
                const parent = this.closest('.accordion');
                if (parent) {
                    parent.querySelectorAll('.accordion-collapse').forEach(collapse => {
                        const targetId = this.getAttribute('data-bs-target')?.replace('#', '');
                        if (collapse.id !== targetId) {
                            collapse.classList.remove('show');
                        }
                    });
                }
            });
        });
    }

    // ---------- Contact Form ----------
    function initContactForm() {
        const form = document.getElementById('contactForm');
        if (!form) return;

        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            // Validate
            if (!data.name || !data.email || !data.message) {
                showToast('Please fill in all fields.', 'error');
                return;
            }

            // Validate email
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(data.email)) {
                showToast('Please enter a valid email address.', 'error');
                return;
            }

            // Simulate sending
            showLoading();
            setTimeout(() => {
                hideLoading();
                showToast('Thank you for your message. We will get back to you soon.', 'success');
                this.reset();
            }, 1500);
        });
    }

    // ---------- Scroll Progress Bar ----------
    function initScrollProgress() {
        const progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress';
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', () => {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = (scrollTop / docHeight) * 100;
            progressBar.style.width = progress + '%';
        });
    }

    // ---------- Tooltips ----------
    function initTooltips() {
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(element => {
                new bootstrap.Tooltip(element);
            });
        }
    }

    // ---------- Form Auto-Save ----------
    function initFormAutoSave() {
        const form = document.getElementById('predictionForm');
        if (!form) return;

        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            // Restore saved data
            const saved = localStorage.getItem('retainiq_form_' + (input.id || input.name));
            if (saved !== null && saved !== 'undefined') {
                input.value = saved;
            }

            // Save on change
            input.addEventListener('change', function() {
                localStorage.setItem('retainiq_form_' + (this.id || this.name), this.value);
            });
        });
    }

    // ---------- Print Button ----------
    function initPrintButton() {
        const printBtn = document.querySelector('[onclick="window.print()"]');
        if (printBtn) {
            printBtn.addEventListener('click', function() {
                window.print();
            });
        }
    }

    // ---------- Console Branding ----------
    console.log('%c RetainIQ v2.0 ', 'background: linear-gradient(135deg, #8B7355, #6B5942); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px;');
    console.log('%c Built with Flask, Bootstrap, and Machine Learning ', 'color: #8B7355; font-size: 12px;');

    // ---------- Expose functions globally ----------
    window.showToast = showToast;
    window.showLoading = showLoading;
    window.hideLoading = hideLoading;

})();