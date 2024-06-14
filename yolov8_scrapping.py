from bing_image_downloader import downloader

downloader.download(
    "zebra in habitat",
    limit=200,
    output_dir="dataset_zebra",
    adult_filter_off=True,
)