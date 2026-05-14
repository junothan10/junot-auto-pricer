document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    // Elements
    const btnSpan = document.querySelector('#predict-btn span');
    const btnLoader = document.getElementById('btn-loader');
    const predictBtn = document.getElementById('predict-btn');
    const resultBox = document.getElementById('result-box');
    const priceOutput = document.getElementById('price-output');
    const summaryTbody = document.getElementById('summary-table-body');

    // Show Loader
    btnSpan.style.display = 'none';
    btnLoader.style.display = 'block';
    predictBtn.disabled = true;
    resultBox.classList.add('hidden');

    // Collect Data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    // Generate Summary Table
    summaryTbody.innerHTML = '';
    
    // Define clean labels for keys
    const labels = {
        'Manufacturer': 'Manufacturer',
        'Vehicle_type': 'Type',
        'Sales_in_thousands': 'Sales (k)',
        '__year_resale_value': 'Resale Value',
        'Engine_size': 'Engine Size (L)',
        'Horsepower': 'Horsepower',
        'Fuel_capacity': 'Fuel Capacity',
        'Fuel_efficiency': 'Fuel Efficiency',
        'Wheelbase': 'Wheelbase',
        'Width': 'Width',
        'Length': 'Length',
        'Curb_weight': 'Curb Weight'
    };

    for (const [key, val] of Object.entries(data)) {
        if (!val) continue; // Skip empty
        const tr = document.createElement('tr');
        const th = document.createElement('th');
        const td = document.createElement('td');
        
        th.textContent = labels[key] || key;
        td.textContent = val;
        
        tr.appendChild(th);
        tr.appendChild(td);
        summaryTbody.appendChild(tr);
    }

    try {
        // Fetch API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Simulate a tiny delay for aesthetic loading effect
        setTimeout(() => {
            if (result.status === 'success') {
                priceOutput.textContent = result.predicted_price_formatted;
                
                // Show Result Box smoothly
                resultBox.classList.remove('hidden');
                
                // Scroll to result smoothly
                setTimeout(() => {
                    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
            } else {
                alert('Error predicting price: ' + result.message);
            }

            // Restore Button
            btnSpan.style.display = 'block';
            btnLoader.style.display = 'none';
            predictBtn.disabled = false;
        }, 800);

    } catch (error) {
        alert('Network error or server is down!');
        btnSpan.style.display = 'block';
        btnLoader.style.display = 'none';
        predictBtn.disabled = false;
    }
});
