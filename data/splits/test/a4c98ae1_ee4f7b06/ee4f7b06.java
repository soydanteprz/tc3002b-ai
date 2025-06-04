import java.io.*;
    import java.util.*;
    import java.lang.Math;


    public class Main {

          static Scanner sc;
          static StringBuilder ans;


          /*-------------- for input in an value ---------------------*/
          static int ni()     { return sc.nextInt();       }
          
          
          




          /*-------------- for input in an array ---------------------*/
          
          
              
          


          /*-------------- File vs Input ---------------------*/
          
          static void runIo() throws Exception  {
               sc = new Scanner(System.in);
          }    

          

          

          
          
           
          
                   

          

          

          
  

          
     
          public static void main(String[] args)  throws Exception {

               // runFile();

               runIo();              

               int t;
               t = 1;     
            
               // t =  sc.nextInt();
          
               ans = new StringBuilder();
          
               while( t-- > 0 ) {   
                    solve();
               }                
          
               System.out.println(ans+""); 
 
          }
    
          static int N ; 
          static int M ; 
          static ArrayList<Integer> f;
          static ArrayList<Integer> e;
          static long dp[][];

          static long find(int i, int j ) {
               
               if( i == N )   return 0;
               if( j == M )   return Integer.MAX_VALUE;


               if (dp[i][j] != -1 )
                    return dp[i][j];

               return dp[i][j] = Math.min( find(i, j+1), Math.abs(f.get(i)-e.get(j)) +  find(i+1, j+1) );
          }       


          public static void solve() {          
          
               int n = ni();
               f = new ArrayList();
               e = new ArrayList();               

               for(int i = 0; i<n; i++) {
                    int v = ni();
                    if( v == 0 ) {
                         e.add(i);
                    }
                    else
                         f.add(i);
               }

               N = f.size();
               M = e.size();               

               dp = new long[N][M];

               for(int i = 0; i<N; i++)
                    Arrays.fill(dp[i], -1);

               ans.append(find(0, 0)+"\n");
          }
 
     }
