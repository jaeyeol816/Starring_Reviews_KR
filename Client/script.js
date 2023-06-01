async function searchReviews() {
	let keyword = document.getElementById('search').value;
	let loading = document.getElementById('loading');
	let avgStar = document.getElementById('avg-star');
	let outputs = document.getElementById('outputs');
	let errorMessage = document.getElementById('error-message');

	avgStar.innerHTML = '';
	outputs.innerHTML = '';
	errorMessage.innerHTML = '';
	loading.style.display = 'block';

	try {
			let response = await fetch('http://localhost:5000/predict', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ keyword: keyword })
			});

			let data = await response.json();
			loading.style.display = 'none';

			if (data.error) {
					errorMessage.innerHTML = data.error;
			} else {
					avgStar.innerHTML = '예측 별점: ' + '★'.repeat(Math.round(data.avg));
					for (let review in data.outputs) {
							let item = document.createElement('p');
							item.innerHTML = `${review}: ${'★'.repeat(Math.round(data.outputs[review]))}`;
							outputs.appendChild(item);
					}
			}
	} catch (error) {
			console.error('Error:', error);
	}
}
