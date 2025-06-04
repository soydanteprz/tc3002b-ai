import java.io.*;
import java.util.*;


public class A{
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Task solver = new Task();
        solver.solve(in, out);
        out.close();
    }
    // main solver
    static class Task{

        double eps= 0.00000001;
        static final int MAXN = 10000001; 
       
        // stores smallest prime factor for every number 
        static int spf[] = new int[MAXN]; 

        Map<Integer,Set<Integer>> dp= new HashMap<>();
        
        // Calculating SPF (Smallest Prime Factor) for every 
        // number till MAXN. 
        // Time Complexity : O(nloglogn) 
         
        
        // A O(log n) function returning primefactorization 
        // by dividing by smallest prime factor at every step 
         
        // function to find first index >= x 
         
        
        // function to find last index <= y 
         
        
        // function to count elements within given range 
        
        ArrayList<Integer>A = new ArrayList<>();
        ArrayList<Integer>B = new ArrayList<>();
        ArrayList<List<Integer>> L = new ArrayList<List<Integer>>();
        public void solve(InputReader in, PrintWriter out) {           
            int n= in.nextInt();
            int[] A= new int[n];
            final int INF= 1000000000;
            List<Integer> pos= new ArrayList<>();
            for (int i = 0; i < n; i++){
                A[i]= in.nextInt();
                if (A[i] == 1) pos.add(i);
            }
            int cnt = pos.size();
            int[][] dp= new int[cnt][n+1];
            for(int[] temp: dp) Arrays.fill(temp,INF);
            if (cnt == 0){
                out.println(0);
                return;
            }
            for (int i = n - 1; i >= 0; i--){
                dp[cnt - 1][i] = dp[cnt - 1][i + 1];
                if (A[i] != 1){
                    dp[cnt - 1][i] = Math.min(dp[cnt - 1][i], Math.abs(i - pos.get(pos.size()-1)));
                }
            }
            for (int i = cnt - 2; i >= 0; i--){
                for (int j = n - 1; j >= 0; j--){
                    dp[i][j] = dp[i][j + 1];
                    if (A[j] != 1){
                        dp[i][j] = Math.min(dp[i][j], Math.abs(j - pos.get(i)) + dp[i + 1][j + 1]);
                    }
                }
            }
            out.println(dp[0][0]);
        }
        
        

    }

    

    
    // fast input reader class;
    
}
