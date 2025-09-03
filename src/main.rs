use regressao_linear::*;

fn main() {
    println!("=== Análise de Regressão Linear ===\n");
    
    // Série temporal de exemplo (y)
    let y = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0];
    
    println!("Dados originais: {:?}\n", y);
    
    // Calcular estatísticas descritivas
    match calcular_estatisticas(&y) {
        Ok(stats) => println!("{}", stats),
        Err(e) => eprintln!("Erro ao calcular estatísticas: {}", e),
    }
    
    // Realizar análise completa de regressão
    match analise_completa(&y) {
        Ok(resultado) => {
            println!("{}", resultado);
            
            // Mostrar valores previstos vs reais
            println!("=== Comparação: Real vs Previsto ===");
            for (i, (&real, &previsto)) in y.iter().zip(resultado.valores_previstos.iter()).enumerate() {
                println!("Período {}: Real = {:.2}, Previsto = {:.2}, Erro = {:.2}", 
                         i, real, previsto, real - previsto);
            }
            
            // Fazer previsões para os próximos períodos
            println!("\n=== Previsões Futuras ===");
            let previsoes = resultado.prever_proximos_periodos(y.len(), 5);
            for (i, previsao) in previsoes.iter().enumerate() {
                println!("Período {}: {:.2}", y.len() + i, previsao);
            }
            
            // Exemplo de previsão para valores específicos de x
            println!("\n=== Previsões para Valores Específicos ===");
            let x_valores = vec![10.0, 15.0, 20.0];
            let previsoes_especificas = resultado.prever(&x_valores);
            for (&x, &pred) in x_valores.iter().zip(previsoes_especificas.iter()) {
                println!("x = {}: y = {:.2}", x, pred);
            }
            
        }
        Err(e) => {
            eprintln!("Erro na análise de regressão: {}", e);
        }
    }
    
    // Exemplo com dados xy personalizados
    println!("\n=== Exemplo com Dados X,Y Personalizados ===");
    let x_custom = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_custom = vec![2.1, 3.9, 6.1, 7.8, 10.2];
    
    match regressao_linear_xy(&x_custom, &y_custom) {
        Ok((a, b)) => {
            println!("Regressão para dados personalizados:");
            println!("y = {:.3}x + {:.3}", a, b);
            
            let y_pred_custom: Vec<f64> = x_custom.iter()
                .map(|&x| a * x + b)
                .collect();
            
            if let Ok(r2) = calcular_r2(&y_custom, &y_pred_custom) {
                println!("R² = {:.4}", r2);
            }
        }
        Err(e) => eprintln!("Erro: {}", e),
    }
}