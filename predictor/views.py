from django.shortcuts import render

# Create your views here.
# predictor/views.py
from django.shortcuts import render
from .forms import UploadFileForm
from .ml import load_data, train_model, evaluate_model
import tempfile
import os

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Save file temporarily
            f = request.FILES['file']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                for chunk in f.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            evidence, labels = load_data(tmp_path)
            model = train_model(evidence, labels)
            acc = evaluate_model(model, evidence, labels)

            os.remove(tmp_path)

            return render(request, 'predictor/result.html', {
                'accuracy': round(acc * 100, 2)
            })
    else:
        form = UploadFileForm()
    return render(request, 'predictor/index.html', {'form': form})

