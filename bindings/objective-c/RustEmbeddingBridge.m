#import "RustEmbeddingBridge.h"
#import "rust_embedding_lib.h"

@implementation RustEmbeddingBridge

+ (void)initModelWithConfigPath:(NSString *)configPath 
                tokenizerPath:(NSString *)tokenizerPath 
                  weightsPath:(NSString *)weightsPath 
             approximateGelu:(BOOL)approximateGelu {
    const char *cConfigPath = [configPath UTF8String];
    const char *cTokenizerPath = [tokenizerPath UTF8String];
    const char *cWeightsPath = [weightsPath UTF8String];
    init_model(cConfigPath, cTokenizerPath, cWeightsPath, approximateGelu);
}

+ (NSArray<NSNumber *> *)generateEmbeddingsFromText:(NSString *)text {
    const char *cText = [text UTF8String];
    EmbeddingResult result = generate_embeddings(cText);

    if (result.error != NULL) {
        NSString *errorString = [NSString stringWithUTF8String:result.error];
        NSLog(@"Error: %@", errorString);
        free_embeddings(result);
        return nil;
    }

    NSMutableArray *embeddingsArray = [NSMutableArray arrayWithCapacity:result.len];
    for (NSUInteger i = 0; i < result.len; i++) {
        [embeddingsArray addObject:@(result.embeddings[i])];
    }

    free_embeddings(result);
    return embeddingsArray;
}

@end

