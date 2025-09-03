#[cfg(test)]
mod tests {
    use super::*;
    
    // Função auxiliar para comparar floats
    fn assert_approx_eq(a: f64, b: f64, precision: f64) {
        assert!((a - b).abs() < precision, "Expected {} ≈ {}, diff: {}", a, b, (a - b).abs());
    }
    
    #[test]
    fn test_regressao_linear_serie_perfeita() {
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (a, b) = regressao_linear(&y).unwrap();
        
        assert_approx_eq(a, 2.0, 0.001); // inclinação
        assert_approx_eq(b, 2.0, 0.001); // intercepto
    }
    
    #[test]
    fn test_regressao_linear_dados_vazios() {
        let y: Vec<f64> = vec![];
        let resultado = regressao_linear(&y);
        
        assert!(matches!(resultado, Err(RegressaoError::DadosVazios)));
    }
    
    #[test]
    fn test_regressao_linear_dados_insuficientes() {
        let y = vec![5.0];
        let resultado = regressao_linear(&y);
        
        assert!(matches!(resultado, Err(RegressaoError::DadosInsuficientes)));
    }
    
    #[test]
    fn test_regressao_linear_xy() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let (a, b) = regressao_linear_xy(&x, &y).unwrap();
        
        assert_approx_eq(a, 2.0, 0.001);
        assert_approx_eq(b, 0.0, 0.001);
    }
    
    #[test]
    fn test_regressao_linear_xy_tamanhos_diferentes() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let resultado = regressao_linear_xy(&x, &y);
        
        assert!(matches!(resultado, Err(RegressaoError::TamanhosDiferentes)));
    }
    
    #[test]
    fn test_calcular_r2_perfeito() {
        let y_real = vec![1.0, 2.0, 3.0, 4.0];
        let y_previsto = vec![1.0, 2.0, 3.0, 4.0];
        let r2 = calcular_r2(&y_real, &y_previsto).unwrap();
        
        assert_approx_eq(r2, 1.0, 0.001);
    }
    
    #[test]
    fn test_calcular_mse() {
        let y_real = vec![1.0, 2.0, 3.0, 4.0];
        let y_previsto = vec![1.1, 1.9, 3.1, 3.9];
        let mse = calcular_mse(&y_real, &y_previsto).unwrap();
        
        // MSE = (0.01 + 0.01 + 0.01 + 0.01) / 4 = 0.01
        assert_approx_eq(mse, 0.01, 0.001);
    }
    
    #[test]
    fn test_calcular_mae() {
        let y_real = vec![1.0, 2.0, 3.0, 4.0];
        let y_previsto = vec![1.1, 1.9, 3.1, 3.9];
        let mae = calcular_mae(&y_real, &y_previsto).unwrap();
        
        // MAE = (0.1 + 0.1 + 0.1 + 0.1) / 4 = 0.1
        assert_approx_eq(mae, 0.1, 0.001);
    }
    
    #[test]
    fn test_prever_valores() {
        let previsoes = prever_valores(5, 3, 2.0, 1.0);
        let esperado = vec![11.0, 13.0, 15.0];
        
        assert_eq!(previsoes.len(), esperado.len());
        for (prev, esp) in previsoes.iter().zip(esperado.iter()) {
            assert_approx_eq(*prev, *esp, 0.001);
        }
    }
    
    #[test]
    fn test_analise_completa() {
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let resultado = analise_completa(&y).unwrap();
        
        assert_approx_eq(resultado.inclinacao, 2.0, 0.001);
        assert_approx_eq(resultado.intercepto, 2.0, 0.001);
        assert_approx_eq(resultado.r_quadrado, 1.0, 0.001);
        assert_approx_eq(resultado.mse, 0.0, 0.001);
    }
    
    #[test]
    fn test_resultado_regressao_prever() {
        let resultado = ResultadoRegressao {
            inclinacao: 2.0,
            intercepto: 1.0,
            r_quadrado: 0.95,
            mse: 0.1,
            rmse: 0.316,
            mae: 0.05,
            valores_previstos: vec![],
        };
        
        let x_valores = vec![0.0, 1.0, 2.0];
        let previsoes = resultado.prever(&x_valores);
        let esperado = vec![1.0, 3.0, 5.0];
        
        for (prev, esp) in previsoes.iter().zip(esperado.iter()) {
            assert_approx_eq(*prev, *esp, 0.001);
        }
    }
    
    #[test]
    fn test_calcular_estatisticas() {
        let dados = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calcular_estatisticas(&dados).unwrap();
        
        assert_approx_eq(stats.media, 3.0, 0.001);
        assert_approx_eq(stats.mediana, 3.0, 0.001);
        assert_approx_eq(stats.minimo, 1.0, 0.001);
        assert_approx_eq(stats.maximo, 5.0, 0.001);
        assert_approx_eq(stats.amplitude, 4.0, 0.001);
    }
    
    #[test]
    fn test_calcular_estatisticas_dados_vazios() {
        let dados: Vec<f64> = vec![];
        let resultado = calcular_estatisticas(&dados);
        
        assert!(matches!(resultado, Err(RegressaoError::DadosVazios)));
    }
    
    #[test]
    fn test_variancia_zero() {
        let x = vec![5.0, 5.0, 5.0, 5.0]; // Todos os valores iguais
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let resultado = regressao_linear_xy(&x, &y);
        
        assert!(matches!(resultado, Err(RegressaoError::VarianciaZero)));
    }
    
    #[test]
    fn test_mediana_par() {
        let dados = vec![1.0, 2.0, 3.0, 4.0];
        let stats = calcular_estatisticas(&dados).unwrap();
        
        assert_approx_eq(stats.mediana, 2.5, 0.001);
    }
    
    #[test]
    fn test_mediana_impar() {
        let dados = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calcular_estatisticas(&dados).unwrap();
        
        assert_approx_eq(stats.mediana, 3.0, 0.001);
    }
    
    #[test]
    fn test_casos_extremos_numericos() {
        // Teste com números muito pequenos
        let y_pequenos = vec![1e-10, 2e-10, 3e-10, 4e-10];
        let resultado = regressao_linear(&y_pequenos);
        assert!(resultado.is_ok());
        
        // Teste com números muito grandes  
        let y_grandes = vec![1e10, 2e10, 3e10, 4e10];
        let resultado = regressao_linear(&y_grandes);
        assert!(resultado.is_ok());
    }
    
    #[test]
    fn test_r2_casos_limite() {
        // R² com ajuste ruim (dados aleatórios)
        let y_real = vec![1.0, 2.0, 3.0, 4.0];
        let y_previsto = vec![4.0, 1.0, 2.0, 3.0]; // Previsões ruins
        let r2 = calcular_r2(&y_real, &y_previsto).unwrap();
        assert!(r2 < 0.5); // R² deve ser baixo
    }
    
    #[test]
    fn test_previsoes_negativas() {
        // Teste com coeficientes que geram valores negativos
        let previsoes = prever_valores(0, 3, -2.0, 5.0);
        let esperado = vec![5.0, 3.0, 1.0];
        
        for (prev, esp) in previsoes.iter().zip(esperado.iter()) {
            assert_approx_eq(*prev, *esp, 0.001);
        }
    }
}