import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pathlib
import gdown
pathlib.Path(f"cache").mkdir(exist_ok=True)
url = 'https://drive.google.com/u/0/uc?id=1pQvg2sT7h9t_srgmN1nGGMfIPa62U9ag'
output = 'model.tar.gz'
gdown.download(url, output, quiet=False)
url = 'https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0'
output = 'spider.zip'
gdown.download(url, output, quiet=False)
