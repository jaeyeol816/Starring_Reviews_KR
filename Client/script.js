async function searchReviews() {
	let keyword = document.getElementById('search').value;
	let loading = document.getElementById('loading');
	let results = document.getElementById('results');
	let avgStar = document.getElementById('avg-star');
	let outputs = document.getElementById('outputs');
	let errorMessage = document.getElementById('error-message');

	avgStar.innerHTML = '';
	outputs.innerHTML = '';
	errorMessage.innerHTML = '';
	loading.style.display = 'block';
	results.style.display = 'none';

	try {
			let response = await fetch('http://34.22.81.10:5000/predict', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ keyword: keyword })
			});

			let data = await response.json();
			loading.style.display = 'none';

			if (data.error) {
					errorMessage.innerHTML = data.error;
			} else {
					avgStar.innerHTML = '예측 별점: ' + '★'.repeat(Math.round(data.avg)) + ' ' + data.avg.toFixed(2);
					for (let review in data.outputs) {
							let item = document.createElement('p');
							item.innerHTML = `<span class="score">${review}: ${data.outputs[review].toFixed(2)}</span>`;
							outputs.appendChild(item);
					}
					results.style.display = 'block';
			}
	} catch (error) {
			console.error('Error:', error);
	}
}

function closeResults() {
	document.getElementById('results').style.display = 'none';
	document.getElementById('search').value = '';
}
