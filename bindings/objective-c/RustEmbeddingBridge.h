#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface RustEmbeddingBridge : NSObject

+ (void)initModelWithConfigPath:(NSString *)configPath 
                tokenizerPath:(NSString *)tokenizerPath 
                  weightsPath:(NSString *)weightsPath 
             approximateGelu:(BOOL)approximateGelu;

+ (NSArray<NSNumber *> *)generateEmbeddingsFromText:(NSString *)text;

@end

NS_ASSUME_NONNULL_END

