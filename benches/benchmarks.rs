

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use regressao_linear::*;

fn benchmark_regressao_pequena(c: &mut Criterion) {
    let dados = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    
    c.bench_function("regressao_pequena", |b| {
        b.iter(|| regressao_linear(black_box(&dados)))
    });
}

fn benchmark_regressao_media(c: &mut Criterion) {
    let dados: Vec<f64> = (0..1000).map(|x| x as f64 * 0.5 + 10.0).collect();
    
    c.bench_function("regressao_media", |b| {
        b.iter(|| regressao_linear(black_box(&dados)))
    });
}

fn benchmark_regressao_grande(c: &mut Criterion) {
    let dados: Vec<f64> = (0..100000).map(|x| x as f64 * 0.001 + 100.0).collect();
    
    c.bench_function("regressao_grande", |b| {
        b.iter(|| regressao_linear(black_box(&dados)))
    });
}

fn benchmark_analise_completa(c: &mut Criterion) {
    let dados: Vec<f64> = (0..1000).map(|x| x as f64 * 0.5 + 10.0 + (x as f64 * 0.01).sin()).collect();
    
    c.bench_function("analise_completa", |b| {
        b.iter(|| analise_completa(black_box(&dados)))
    });
}

fn benchmark_calcular_r2(c: &mut Criterion) {
    let y_real: Vec<f64> = (0..1000).map(|x| x as f64 * 0.5 + 10.0).collect();
    let y_prev: Vec<f64> = (0..1000).map(|x| x as f64 * 0.5 + 10.1).collect();
    
    c.bench_function("calcular_r2", |b| {
        b.iter(|| calcular_r2(black_box(&y_real), black_box(&y_prev)))
    });
}

fn benchmark_estatisticas(c: &mut Criterion) {
    let dados: Vec<f64> = (0..10000).map(|x| x as f64 * 0.1 + (x as f64 * 0.01).sin()).collect();
    
    c.bench_function("calcular_estatisticas", |b| {
        b.iter(|| calcular_estatisticas(black_box(&dados)))
    });
}

criterion_group!(
    benches,
    benchmark_regressao_pequena,
    benchmark_regressao_media,
    benchmark_regressao_grande,
    benchmark_analise_completa,
    benchmark_calcular_r2,
    benchmark_estatisticas
);

criterion_main!(benches);