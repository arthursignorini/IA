import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class CustomVisionClient {
    public static void main(String[] args) {
        // Configurações da API
        String predictionUrl = "https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/dc06014f-a870-4eed-a1fc-b89a273311e6/classify/iterations/MatchPoint/image";
        String predictionKey = "7f7e301ecf9247798b05bc9f12567333";
        
        // Caminho da imagem que será enviada para a classificação
        Path imagePath = Paths.get("volleyball/5.jpg");

        try {
            // Criação do cliente HTTP
            HttpClient client = HttpClient.newHttpClient();

            // Carregar a imagem como um array de bytes
            byte[] imageBytes = Files.readAllBytes(imagePath);

            // Construção da requisição HTTP POST
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(predictionUrl))
                .header("Content-Type", "application/octet-stream")
                .header("Prediction-Key", predictionKey)
                .POST(HttpRequest.BodyPublishers.ofByteArray(imageBytes))
                .build();

            // Envio da requisição e obtenção da resposta
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            String responseBody = response.body();

            // Exibir a resposta da API
            System.out.println("Resposta da API:");
            // System.out.println(responseBody);
            System.out.println("-------------------------------");

            // Processar a resposta manualmente
            parseResponse(responseBody);
        } catch (Exception e) {
            System.out.println("Erro ao enviar a imagem: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void parseResponse(String responseBody) {
        // Extrai o conteúdo dentro de "predictions"
        String predictionsSection = responseBody.split("\"predictions\":\\[")[1];
        predictionsSection = predictionsSection.split("]")[0];

        // Divide cada predição
        String[] predictions = predictionsSection.split("\\},\\{");

        String topSport = null;
        double topProbability = 0.0;

        System.out.println("Resultado da Classificação:");
        System.out.println("-------------------------------");

        // Encontra o esporte com a maior probabilidade
        for (String prediction : predictions) {
            // Extrai o valor de "tagName"
            String tagName = extractValue(prediction, "\"tagName\":\"", "\"");

            // Extrai o valor de "probability"
            String probabilityString = extractValue(prediction, "\"probability\":", ",");
            double probability = Double.parseDouble(probabilityString) * 100; // Convertendo para porcentagem

            System.out.printf("Esporte: %s - Probabilidade: %.2f%%\n", tagName, probability);

            // Atualiza o esporte com a maior probabilidade
            if (probability > topProbability) {
                topSport = tagName;
                topProbability = probability;
            }
        }

        System.out.println("-------------------------------");

        // Exibe as regras do esporte com a maior probabilidade
        if (topSport != null) {
            printSportRules(topSport);
        } else {
            System.out.println("Nenhum esporte identificado com probabilidade suficiente.");
        }
    }

    public static String extractValue(String source, String startDelimiter, String endDelimiter) {
        int start = source.indexOf(startDelimiter) + startDelimiter.length();
        int end = source.indexOf(endDelimiter, start);
        return source.substring(start, end);
    }

    // Método para imprimir as regras de cada esporte
    public static void printSportRules(String sport) {
        Map<String, String> sportRules = new HashMap<>();
        sportRules.put("Basquete", "Regras do Basquete:\n1. Cada equipe tem 5 jogadores em quadra.\n2. A cesta vale 2 pontos ou 3 pontos, dependendo da distância.\n3. O jogo é dividido em 4 períodos de 10 ou 12 minutos.");
        sportRules.put("Futebol", "Regras do Futebol:\n1. Cada equipe tem 11 jogadores em campo.\n2. O jogo tem dois tempos de 45 minutos cada.\n3. O objetivo é marcar gols na baliza adversária.");
        sportRules.put("Vôlei", "Regras do Voleibol:\n1. Cada equipe tem 6 jogadores em quadra.\n2. O objetivo é fazer a bola tocar o chão do lado adversário.\n3. O jogo é disputado em sets de 25 pontos.");

        String rules = sportRules.getOrDefault(sport, "Regras não disponíveis para este esporte.");
        System.out.printf("Regras do esporte com maior probabilidade (%s):\n%s\n", sport, rules);
    }
}
