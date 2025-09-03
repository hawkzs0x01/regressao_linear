//! # Biblioteca de Regressão Linear
//! 
//! Esta biblioteca fornece funcionalidades para análise de regressão linear,
//! incluindo cálculo de coeficientes, métricas de avaliação e previsões.


use std::fmt;

/// Erro personalizado para operações de regressão linear
#[derive(Debug, Clone, PartialEq)]
pub enum RegressaoError {
    DadosInsuficientes,
    DadosVazios,
    VarianciaZero,
    TamanhosDiferentes,
}

impl fmt::Display for RegressaoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RegressaoError::DadosInsuficientes => write!(f, "Dados insuficientes para regressão"),
            RegressaoError::DadosVazios => write!(f, "Conjunto de dados vazio"),
            RegressaoError::VarianciaZero => write!(f, "Variância zero nos dados"),
            RegressaoError::TamanhosDiferentes => write!(f, "Vetores com tamanhos diferentes"),
        }
    }
}

impl std::error::Error for RegressaoError {}

/// Tipo Result personalizado para esta biblioteca
pub type Resultado<T> = Result<T, RegressaoError>;

/// Estrutura para armazenar resultados da análise de regressão
#[derive(Debug, Clone)]
pub struct ResultadoRegressao {
    pub inclinacao: f64,
    pub intercepto: f64,
    pub r_quadrado: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub valores_previstos: Vec<f64>,
}

impl ResultadoRegressao {
    /// Faz previsões para novos valores de x
    pub fn prever(&self, x_valores: &[f64]) -> Vec<f64> {
        x_valores.iter()
            .map(|&x| self.inclinacao * x + self.intercepto)
            .collect()
    }
    
    /// Faz previsões para os próximos n períodos (série temporal)
    pub fn prever_proximos_periodos(&self, inicio: usize, n_periodos: usize) -> Vec<f64> {
        (inicio..inicio + n_periodos)
            .map(|x| self.inclinacao * x as f64 + self.intercepto)
            .collect()
    }
}

impl fmt::Display for ResultadoRegressao {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "=== Resultado da Regressão Linear ===")?;
        writeln!(f, "Inclinação (a): {:.6}", self.inclinacao)?;
        writeln!(f, "Intercepto (b): {:.6}", self.intercepto)?;
        writeln!(f, "R²: {:.6}", self.r_quadrado)?;
        writeln!(f, "MSE: {:.6}", self.mse)?;
        writeln!(f, "RMSE: {:.6}", self.rmse)?;
        writeln!(f, "MAE: {:.6}", self.mae)?;
        Ok(())
    }
}

/// Calcula a regressão linear para uma série temporal (x implícito como índices)
/// 
/// # Argumentos
/// * `y` - Vetor com os valores y da série temporal
/// 
/// # Retorna
/// * `Ok((inclinacao, intercepto))` - Os coeficientes da regressão
/// * `Err(RegressaoError)` - Em caso de erro
pub fn regressao_linear(y: &[f64]) -> Resultado<(f64, f64)> {
    if y.is_empty() {
        return Err(RegressaoError::DadosVazios);
    }
    
    if y.len() < 2 {
        return Err(RegressaoError::DadosInsuficientes);
    }
    
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    
    regressao_linear_xy(&x, y)
}

/// Calcula a regressão linear para pontos (x, y) arbitrários
/// 
/// # Argumentos
/// * `x` - Vetor com os valores x
/// * `y` - Vetor com os valores y
/// 
/// # Retorna
/// * `Ok((inclinacao, intercepto))` - Os coeficientes da regressão
/// * `Err(RegressaoError)` - Em caso de erro
pub fn regressao_linear_xy(x: &[f64], y: &[f64]) -> Resultado<(f64, f64)> {
    if x.is_empty() || y.is_empty() {
        return Err(RegressaoError::DadosVazios);
    }
    
    if x.len() != y.len() {
        return Err(RegressaoError::TamanhosDiferentes);
    }
    
    if x.len() < 2 {
        return Err(RegressaoError::DadosInsuficientes);
    }
    
    let n = x.len() as f64;
    
    // Calcular médias
    let media_x = x.iter().sum::<f64>() / n;
    let media_y = y.iter().sum::<f64>() / n;
    
    // Calcular somatórias para os coeficientes
    let mut soma_xy = 0.0;
    let mut soma_xx = 0.0;
    
    for i in 0..x.len() {
        let diff_x = x[i] - media_x;
        let diff_y = y[i] - media_y;
        soma_xy += diff_x * diff_y;
        soma_xx += diff_x * diff_x;
    }
    
    // Verificar se há variância em x
    if soma_xx.abs() < f64::EPSILON {
        return Err(RegressaoError::VarianciaZero);
    }
    
    // Calcular coeficientes
    let inclinacao = soma_xy / soma_xx;
    let intercepto = media_y - inclinacao * media_x;
    
    Ok((inclinacao, intercepto))
}

/// Realiza análise completa de regressão linear
pub fn analise_completa(y: &[f64]) -> Resultado<ResultadoRegressao> {
    let (inclinacao, intercepto) = regressao_linear(y)?;
    
    // Calcular valores previstos
    let valores_previstos: Vec<f64> = (0..y.len())
        .map(|x| inclinacao * x as f64 + intercepto)
        .collect();
    
    // Calcular métricas
    let r_quadrado = calcular_r2(y, &valores_previstos)?;
    let mse = calcular_mse(y, &valores_previstos)?;
    let rmse = mse.sqrt();
    let mae = calcular_mae(y, &valores_previstos)?;
    
    Ok(ResultadoRegressao {
        inclinacao,
        intercepto,
        r_quadrado,
        mse,
        rmse,
        mae,
        valores_previstos,
    })
}

/// Calcula o R² (coeficiente de determinação)
pub fn calcular_r2(y_real: &[f64], y_previsto: &[f64]) -> Resultado<f64> {
    if y_real.is_empty() || y_previsto.is_empty() {
        return Err(RegressaoError::DadosVazios);
    }
    
    if y_real.len() != y_previsto.len() {
        return Err(RegressaoError::TamanhosDiferentes);
    }
    
    let media_y = y_real.iter().sum::<f64>() / y_real.len() as f64;
    
    let mut ss_tot = 0.0; // Soma total dos quadrados
    let mut ss_res = 0.0; // Soma residual dos quadrados
    
    for i in 0..y_real.len() {
        ss_tot += (y_real[i] - media_y).powi(2);
        ss_res += (y_real[i] - y_previsto[i]).powi(2);
    }
    
    if ss_tot.abs() < f64::EPSILON {
        return Err(RegressaoError::VarianciaZero);
    }
    
    Ok(1.0 - (ss_res / ss_tot))
}

/// Calcula o MSE (Mean Squared Error)
pub fn calcular_mse(y_real: &[f64], y_previsto: &[f64]) -> Resultado<f64> {
    if y_real.is_empty() || y_previsto.is_empty() {
        return Err(RegressaoError::DadosVazios);
    }
    
    if y_real.len() != y_previsto.len() {
        return Err(RegressaoError::TamanhosDiferentes);
    }
    
    let soma_erros_quadrados: f64 = y_real.iter()
        .zip(y_previsto.iter())
        .map(|(real, prev)| (real - prev).powi(2))
        .sum();
    
    Ok(soma_erros_quadrados / y_real.len() as f64)
}

/// Calcula o MAE (Mean Absolute Error)
pub fn calcular_mae(y_real: &[f64], y_previsto: &[f64]) -> Resultado<f64> {
    if y_real.is_empty() || y_previsto.is_empty() {
        return Err(RegressaoError::DadosVazios);
    }
    
    if y_real.len() != y_previsto.len() {
        return Err(RegressaoError::TamanhosDiferentes);
    }
    
    let soma_erros_absolutos: f64 = y_real.iter()
        .zip(y_previsto.iter())
        .map(|(real, prev)| (real - prev).abs())
        .sum();
    
    Ok(soma_erros_absolutos / y_real.len() as f64)
}

/// Faz previsões para valores futuros em uma série temporal
pub fn prever_valores(inicio: usize, n_valores: usize, inclinacao: f64, intercepto: f64) -> Vec<f64> {
    (inicio..inicio + n_valores)
        .map(|x| inclinacao * x as f64 + intercepto)
        .collect()
}

/// Calcula estatísticas descritivas básicas
#[derive(Debug, Clone)]
pub struct EstatisticasDescritivas {
    pub media: f64,
    pub mediana: f64,
    pub desvio_padrao: f64,
    pub variancia: f64,
    pub minimo: f64,
    pub maximo: f64,
    pub amplitude: f64,
}

impl fmt::Display for EstatisticasDescritivas {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "=== Estatísticas Descritivas ===")?;
        writeln!(f, "Média: {:.6}", self.media)?;
        writeln!(f, "Mediana: {:.6}", self.mediana)?;
        writeln!(f, "Desvio Padrão: {:.6}", self.desvio_padrao)?;
        writeln!(f, "Variância: {:.6}", self.variancia)?;
        writeln!(f, "Mínimo: {:.6}", self.minimo)?;
        writeln!(f, "Máximo: {:.6}", self.maximo)?;
        writeln!(f, "Amplitude: {:.6}", self.amplitude)?;
        Ok(())
    }
}

/// Calcula estatísticas descritivas de um conjunto de dados
pub fn calcular_estatisticas(dados: &[f64]) -> Resultado<EstatisticasDescritivas> {
    if dados.is_empty() {
        return Err(RegressaoError::DadosVazios);
    }
    
    let n = dados.len() as f64;
    let media = dados.iter().sum::<f64>() / n;
    
    let mut dados_ordenados = dados.to_vec();
    dados_ordenados.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mediana = if dados_ordenados.len() % 2 == 0 {
        let meio = dados_ordenados.len() / 2;
        (dados_ordenados[meio - 1] + dados_ordenados[meio]) / 2.0
    } else {
        dados_ordenados[dados_ordenados.len() / 2]
    };
    
    let variancia = dados.iter()
        .map(|x| (x - media).powi(2))
        .sum::<f64>() / n;
    
    let desvio_padrao = variancia.sqrt();
    let minimo = dados_ordenados[0];
    let maximo = dados_ordenados[dados_ordenados.len() - 1];
    let amplitude = maximo - minimo;
    
    Ok(EstatisticasDescritivas {
        media,
        mediana,
        desvio_padrao,
        variancia,
        minimo,
        maximo,
        amplitude,
    })
}

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