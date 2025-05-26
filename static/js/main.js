document.addEventListener('DOMContentLoaded', () => {
    const continentSelect = document.getElementById('continent');
    const citySelect      = document.getElementById('city');
    const checkboxes      = Array.from(document.querySelectorAll('.category-checkbox'));
    const submitBtn       = document.getElementById('submit-btn');
  
    // 1) Populate city dropdown when continent changes
    continentSelect.addEventListener('change', () => {
      const cont = continentSelect.value;
      const cities = COUNTRY_MAP[cont] || [];
  
      // Clear & enable city select
      citySelect.innerHTML = '<option value="" disabled selected>Select country</option>';
      citySelect.disabled = false;
  
      cities.forEach(country => {
        const opt = document.createElement('option');
        opt.value = country;
        opt.textContent = country;
        citySelect.appendChild(opt);
      });
    });
  
    // 2) Enable submit only if ≥3 checkboxes are checked
    function updateSubmitState() {
      const checkedCount = checkboxes.filter(cb => cb.checked).length;
      submitBtn.disabled = checkedCount < 3;
    }
  
    checkboxes.forEach(cb => {
      cb.addEventListener('change', updateSubmitState);
    });
  
    // Initialize state on page load
    citySelect.disabled = true;
    updateSubmitState();
  });  