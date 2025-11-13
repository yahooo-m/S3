import os
import requests
import hashlib
from tqdm import tqdm


_links = [
    "I will provide the pretrained weight soon"
]

def download_models_if_needed():
    os.makedirs('weights', exist_ok=True)
    for link, md5 in _links:
        # download file if not exists with a progressbar
        filename = link.split('/')[-1]
        if not os.path.exists(os.path.join('weights', filename)) or hashlib.md5(open(os.path.join('weights', filename), 'rb').read()).hexdigest() != md5:
            print(f'Downloading {filename}...')
            r = requests.get(link, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(os.path.join('weights', filename), 'wb') as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise RuntimeError('Error while downloading %s' % filename)
            

if __name__ == '__main__':
    download_models_if_needed()
