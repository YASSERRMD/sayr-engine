use sayr_engine::{CohereClient, LanguageModel};

#[tokio::test]
async fn test_cohere_client_instantiation() {
    let client = CohereClient::new("test-key").with_model("command-light");
    // Verify it implements LanguageModel trait (by creating a trait object)
    let _: Box<dyn LanguageModel> = Box::new(client);
    
    // We can't verify HTTP calls without a mock server or key, 
    // but this confirms the struct definition and trait implementation compile.
}
