//! Exemplo de análise completa com múltiplos datasets

use regressao_linear::*;

fn main() {
    println!("=== Análise Completa de Regressão Linear ===\n");
    
    // Dataset 1: Crescimento populacional de uma cidade
    println!("📈 Dataset 1: População da cidade (em milhares)");
    let populacao = vec![50.0, 52.5, 55.2, 57.8, 60.5, 63.1, 65.8, 68.2];
    analisar_dataset("População", &populacao, "habitantes (milhares)");
    
    println!("\n{}", "=".repeat(60));
    
    // Dataset 2: Temperatura ao longo do dia
    println!("🌡️  Dataset 2: Temperatura ao longo do dia");
    let temperatura = vec![15.0, 18.2, 22.5, 26.8, 30.1, 32.5, 29.8, 25.2];
    analisar_dataset("Temperatura", &temperatura, "°C");
    
    println!("\n{}", "=".repeat(60));
    
    // Dataset 3: Comparação de dois métodos (XY personalizado)
    println!("🔬 Dataset 3: Relação entre horas de estudo e nota");
    let horas_estudo = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let notas = vec![5.2, 6.1, 6.8, 7.5, 8.1, 8.7, 9.2, 9.5];
    
    analisar_dataset_xy(&horas_estudo, &notas, "Horas de Estudo", "Nota");
    
    println!("\n{}", "=".repeat(60));
    
    // Dataset 4: Dados com ruído para testar robustez
    println!("📊 Dataset 4: Dados com ruído (vendas com sazonalidade)");
    let vendas_ruidosas = vec![100.0, 95.0, 130.0, 125.0, 160.0, 155.0, 190.0, 185.0, 220.0];
    analisar_dataset("Vendas com ruído", &vendas_ruidosas, "unidades");
}

fn analisar_dataset(nome: &str, dados: &[f64], unidade: &str) {
    println!("Dados de {}: {:?}", nome, dados);
    
    // Estatísticas descritivas
    match calcular_estatisticas(dados) {
        Ok(stats) => {
            println!("\n📋 Estatísticas Descritivas:");
            println!("   Média: {:.2} {}", stats.media, unidade);
            println!("   Mediana: {:.2} {}", stats.mediana, unidade);
            println!("   Desvio Padrão: {:.2} {}", stats.desvio_padrao, unidade);
            println!("   Min/Max: {:.2} / {:.2} {}", stats.minimo, stats.maximo, unidade);
            println!("   Amplitude: {:.2} {}", stats.amplitude, unidade);
        }
        Err(e) => println!("Erro ao calcular estatísticas: {}", e),
    }
    
    // Análise de regressão completa
    match analise_completa(dados) {
        Ok(resultado) => {
            println!("\n📈 Análise de Regressão:");
            println!("   Equação: y = {:.3}x + {:.3}", resultado.inclinacao, resultado.intercepto);
            println!("   R²: {:.4} ({})", resultado.r_quadrado, interpretar_r2(resultado.r_quadrado));
            println!("   MSE: {:.4}", resultado.mse);
            println!("   RMSE: {:.4}", resultado.rmse);
            println!("   MAE: {:.4}", resultado.mae);
            
            // Mostrar alguns valores previstos vs reais
            println!("\n🎯 Precisão das Previsões (primeiros 5 pontos):");
            for (i, (&real, &prev)) in dados.iter().zip(resultado.valores_previstos.iter()).take(5).enumerate() {
                let erro_percentual = ((real - prev).abs() / real * 100.0).min(999.9);
                println!("   Período {}: Real={:.1}, Prev={:.1}, Erro={:.1}%", 
                         i, real, prev, erro_percentual);
            }
            
            // Previsões futuras
            println!("\n🔮 Previsões para próximos 3 períodos:");
            let previsoes = resultado.prever_proximos_periodos(dados.len(), 3);
            for (i, &valor) in previsoes.iter().enumerate() {
                println!("   Período {}: {:.2} {}", dados.len() + i, valor, unidade);
            }
            
            // Análise de tendência
            if resultado.inclinacao > 0.1 {
                println!("\n📊 Tendência: CRESCENTE (+{:.2} {} por período)", resultado.inclinacao, unidade);
            } else if resultado.inclinacao < -0.1 {
                println!("\n📊 Tendência: DECRESCENTE ({:.2} {} por período)", resultado.inclinacao, unidade);
            } else {
                println!("\n📊 Tendência: ESTÁVEL (variação mínima)");
            }
        }
        Err(e) => println!("Erro na análise: {}", e),
    }
}

fn analisar_dataset_xy(x: &[f64], y: &[f64], nome_x: &str, nome_y: &str) {
    println!("Dados de {} vs {}", nome_x, nome_y);
    println!("{}: {:?}", nome_x, x);
    println!("{}: {:?}", nome_y, y);
    
    match regressao_linear_xy(x, y) {
        Ok((inclinacao, intercepto)) => {
            println!("\n📈 Regressão Linear:");
            println!("   {} = {:.3} × {} + {:.3}", nome_y, inclinacao, nome_x, intercepto);
            
            // Calcular métricas
            let y_prev: Vec<f64> = x.iter().map(|&xi| inclinacao * xi + intercepto).collect();
            
            if let Ok(r2) = calcular_r2(y, &y_prev) {
                println!("   R²: {:.4} ({})", r2, interpretar_r2(r2));
                println!("   Correlação: {}", interpretar_correlacao(inclinacao, r2));
            }
            
            if let Ok(mse) = calcular_mse(y, &y_prev) {
                println!("   RMSE: {:.4}", mse.sqrt());
            }
            
            // Interpretação prática
            if inclinacao > 0.0 {
                println!("\n💡 Interpretação: Cada unidade adicional de {} resulta em +{:.2} de {}", 
                         nome_x, inclinacao, nome_y);
            } else {
                println!("\n💡 Interpretação: Cada unidade adicional de {} resulta em {:.2} de {}", 
                         nome_x, inclinacao, nome_y);
            }
        }
        Err(e) => println!("Erro: {}", e),
    }
}

fn interpretar_r2(r2: f64) -> &'static str {
    match r2 {
        r if r >= 0.9 => "Excelente ajuste",
        r if r >= 0.8 => "Bom ajuste",
        r if r >= 0.7 => "Ajuste moderado",
        r if r >= 0.5 => "Ajuste fraco",
        _ => "Ajuste muito fraco"
    }
}

fn interpretar_correlacao(inclinacao: f64, r2: f64) -> String {
    let forca = if r2 >= 0.8 { "forte" } 
               else if r2 >= 0.5 { "moderada" } 
               else { "fraca" };
    
    let direcao = if inclinacao > 0.0 { "positiva" } else { "negativa" };
    
    format!("Correlação {} {}", forca, direcao)
}