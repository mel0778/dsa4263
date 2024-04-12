import os
import sys
import requests
import tarfile
from tqdm import tqdm


def download_and_extract(url, output_dir, foldername):
    # Download the file
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(output_dir, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=output_dir, file=sys.stdout,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                progress_bar.update(len(data))

    # Extract the tar.bz2 file
    with tarfile.open(output_dir, 'r:bz2') as tar:
        files = tar.getmembers()
        progress_bar = tqdm(files, desc='Extracting', file=sys.stdout)
        for member in progress_bar:
            member.path = os.path.join(
                "..", "..", "data", "raw_external", foldername, os.path.basename(member.path))
            tar.extract(member)

    # Remove the downloaded file
    os.remove(output_dir)


def main():
    download_url_answers = 'https://kilthub.cmu.edu/ndownloader/files/24857828'
    output_filename_answers = 'answers.tar.bz2'
    foldername_answers = 'answers'
    download_and_extract(download_url_answers,
                         output_filename_answers, foldername_answers)

    download_url_r52 = 'https://kilthub.cmu.edu/ndownloader/files/24849938'
    output_filename_r52 = 'r5.2.tar.bz2'
    foldername_r52 = 'r52'
    download_and_extract(download_url_r52, output_filename_r52, foldername_r52)


if __name__ == "__main__":
    main()
