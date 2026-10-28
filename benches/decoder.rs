use std::{fs, io::Cursor};

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion, Throughput,
};
use png::{BitDepth, ColorType, Decoder, Encoder, Reader, Transformations};

#[path = "../src/test_utils.rs"]
mod test_utils;

fn load_all(c: &mut Criterion) {
    let mut g = c.benchmark_group("decode");
    for entry in fs::read_dir("tests/benches/").unwrap().flatten() {
        match entry.path().extension() {
            Some(st) if st == "png" => {}
            _ => continue,
        }

        let data = fs::read(entry.path()).unwrap();
        bench_file(&mut g, data, entry.file_name().into_string().unwrap());
    }
    g.finish();

    // The transformations of a caller that wants 8-bit samples with alpha, whatever the source.
    let mut g = c.benchmark_group("expand-alpha");
    for name in [
        "kodim23.png",
        "lorem_ipsum_screenshot.png",
        "lorem_ipsum_oxipng.png",
        "tango-icon-address-book-new-128-rsvg-convert.png",
    ] {
        let data = fs::read(format!("tests/benches/{name}")).unwrap();
        bench_file_with(&mut g, data, name.to_owned(), EXPAND_ALPHA);
    }
    let kodim23 = fs::read("tests/benches/kodim23.png").unwrap();
    bench_file_with(
        &mut g,
        with_rgb_trns(&kodim23),
        "kodim23-trns.png".to_owned(),
        EXPAND_ALPHA,
    );
    g.finish();

    // Small IDATS
    let mut g = c.benchmark_group("generated-noncompressed-4k-idat");
    bench_noncompressed_png(&mut g, 8, 4096); // 256 B
    bench_noncompressed_png(&mut g, 128, 4096); // 64 KB
    bench_noncompressed_png(&mut g, 2048, 4096); // 16 MB
    bench_noncompressed_png(&mut g, 12288, 4096); // 576 MB
    g.finish();

    // Normal IDATS
    let mut g = c.benchmark_group("generated-noncompressed-64k-idat");
    bench_noncompressed_png(&mut g, 128, 65536); // 64 KB
    bench_noncompressed_png(&mut g, 2048, 65536); // 16 MB
    bench_noncompressed_png(&mut g, 12288, 65536); // 576 MB
    g.finish();

    // Large IDATS
    let mut g = c.benchmark_group("generated-noncompressed-2g-idat");
    bench_noncompressed_png(&mut g, 2048, 0x7fffffff); // 16 MB
    bench_noncompressed_png(&mut g, 12288, 0x7fffffff); // 576 MB
    g.finish();

    // Incremental decoding via `read_row`
    let mut g = c.benchmark_group("row-by-row");
    let mut data = Vec::new();
    test_utils::write_noncompressed_png(&mut data, 128, 4096);
    bench_read_row(&mut g, data, "128x128-4k-idat");
    g.finish();
}

criterion_group! {benches, load_all}
criterion_main!(benches);

fn bench_noncompressed_png(g: &mut BenchmarkGroup<WallTime>, size: u32, idat_bytes: usize) {
    let mut data = Vec::new();
    test_utils::write_noncompressed_png(&mut data, size, idat_bytes);
    bench_file(g, data, format!("{size}x{size}.png"));
}

/// This benchmarks decoding via a call to `Reader::next_frame`.
fn bench_file(g: &mut BenchmarkGroup<WallTime>, data: Vec<u8>, name: String) {
    bench_file_with(g, data, name, Transformations::EXPAND);
}

const EXPAND_ALPHA: Transformations = Transformations::EXPAND
    .union(Transformations::STRIP_16)
    .union(Transformations::ALPHA);

fn bench_file_with(
    g: &mut BenchmarkGroup<WallTime>,
    data: Vec<u8>,
    name: String,
    transformations: Transformations,
) {
    if data.len() > 1_000_000 {
        g.sample_size(10);
    }

    let mut reader = create_reader_with(data.as_slice(), transformations);
    let mut image = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut image).unwrap();

    g.throughput(Throughput::Bytes(info.buffer_size() as u64));
    g.bench_with_input(name, &data, |b, data| {
        b.iter(|| {
            let mut reader = create_reader_with(data.as_slice(), transformations);
            reader.next_frame(&mut image).unwrap();
        })
    });
}

/// This benchmarks decoding via a sequence of `Reader::read_row` calls.
fn bench_read_row(g: &mut BenchmarkGroup<WallTime>, data: Vec<u8>, name: &str) {
    let reader = create_reader(data.as_slice());
    let mut image = vec![0; reader.output_buffer_size().unwrap()];
    let bytes_per_row = reader.output_line_size(reader.info().width).unwrap();
    g.throughput(Throughput::Bytes(image.len() as u64));
    g.bench_with_input(name, &data, |b, data| {
        b.iter(|| {
            let mut reader = create_reader(data.as_slice());

            for output_row in image.chunks_exact_mut(bytes_per_row) {
                reader.read_row(output_row).unwrap().unwrap();
            }
        })
    });
}

fn create_reader(data: &[u8]) -> Reader<Cursor<&[u8]>> {
    // Cover default transformations used by the `image` crate when constructing
    // `image::codecs::png::PngDecoder`.
    create_reader_with(data, Transformations::EXPAND)
}

fn create_reader_with(data: &[u8], transformations: Transformations) -> Reader<Cursor<&[u8]>> {
    let mut decoder = Decoder::new(Cursor::new(data));
    decoder.set_transformations(transformations);
    decoder.read_info().unwrap()
}

/// Re-encodes an 8-bit RGB PNG with a `tRNS` key of its first pixel's color.
fn with_rgb_trns(rgb_png: &[u8]) -> Vec<u8> {
    let mut reader = create_reader_with(rgb_png, Transformations::IDENTITY);
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut pixels).unwrap();
    let (width, height) = reader.info().size();
    assert_eq!(reader.info().color_type, ColorType::Rgb);

    let trns: Vec<u8> = pixels[..3].iter().flat_map(|&sample| [0, sample]).collect();
    let mut encoded = Vec::new();
    let mut encoder = Encoder::new(&mut encoded, width, height);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_trns(trns);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
    encoded
}
