//! Exemplo básico de uso da biblioteca de regressão linear

use regressao_linear::*;

fn main() {
    println!("=== Exemplo Básico de Regressão Linear ===\n");
    
    // Dados de vendas mensais (exemplo fictício)
    let vendas = vec![100.0, 120.0, 140.0, 160.0, 180.0, 200.0];
    
    println!("Dados de vendas mensais: {:?}", vendas);
    
    // Calcular regressão linear simples
    match regressao_linear(&vendas) {
        Ok((inclinacao, intercepto)) => {
            println!("\n=== Resultados ===");
            println!("Equação da reta: y = {:.2}x + {:.2}", inclinacao, intercepto);
            println!("Crescimento mensal: {:.2} unidades", inclinacao);
            
            // Calcular valores previstos
            let previstos: Vec<f64> = (0..vendas.len())
                .map(|x| inclinacao * x as f64 + intercepto)
                .collect();
            
            // Calcular R²
            if let Ok(r2) = calcular_r2(&vendas, &previstos) {
                println!("R² (qualidade do ajuste): {:.4}", r2);
            }
            
            // Prever próximos 3 meses
            let futuras = prever_valores(vendas.len(), 3, inclinacao, intercepto);
            println!("\n=== Previsões para os próximos 3 meses ===");
            for (i, valor) in futuras.iter().enumerate() {
                println!("Mês {}: {:.1} vendas", vendas.len() + i + 1, valor);
            }
            
        }
        Err(e) => {
            eprintln!("Erro: {}", e);
        }
    }
}