import javax.swing.text.html.parser.Entity;
import java.io.*;
import java.util.*;
import java.math.BigInteger;


public class Main{
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
        static final int MAXN = 1010;
        static final int MOD= 1000000007;

        // stores smallest prime factor for every number
        static int spf[] = new int[MAXN];

        static boolean[] prime;

        // Calculating SPF (Smallest Prime Factor) for every
        // number till MAXN.
        // Time Complexity : O(nloglogn)
        

        

        // function to find first index >= x
        
        

        // function to find last index <= y
        
        

        // function to count elements within given range
        

        

        public int add(int a, int b){
            a+=b;
            if(a>=MOD) a-=MOD;
            else if(a<0) a+=MOD;
            return a;
        }

        

        

        int[] fact= new int[MAXN];
        int[] inv= new int[MAXN];

        

        

        

        /**
         * return VALUE of lower bound for unsorted array
         */
        
        /**
         * return VALUE of upper bound for unsorted array
         */
        

        

        

        InputReader in; PrintWriter out;
        static int mod = 1000000007;
        int ans = 0;
        ArrayList<Integer>g[] = new ArrayList[101];
        int cnt[][] = new int[101][101];
        int dp[][] = new int[101][101];
        
        
        
        public void solve(InputReader in, PrintWriter out) {
           int t = in.nextInt();
           while (t-- > 0){
               int n = in.nextInt();
               String[] str = new String[n];
               ArrayList<Integer> diff[] = new ArrayList[5];
               for (int i = 0; i < 5; i++) diff[i] = new ArrayList<>();
               for (int i = 0; i < n; i++){
                   str[i] = in.nextToken();
                   int[] cnt = new int[5];
                   for (int j = 0; j < str[i].length(); j++){
                       cnt[str[i].charAt(j) - 'a']++;
                   }
                   for (int j = 0; j < 5; j++){
                       diff[j].add(cnt[j] * 2 - str[i].length());
                   }
               }
               int ans = 0;
               for (int i = 0; i < 5; i++){
                   Collections.sort(diff[i]);
                   Collections.reverse(diff[i]);
                   int cur = 0, x = 0;
                   for (int j = 0; j < diff[i].size(); j++){
                       cur+=diff[i].get(j);
                       if (cur <= 0){
                           break;
                       }
                       x++;
                   }
                   ans = Math.max(ans, x);
               }
               out.println(ans);
           }
        }

        

    }

    



    

    

    // public static class compareL implements Comparator<Tuple>{
    //     @Override
    //     public int compare(Tuple t1, Tuple t2) {
    //         return t2.l - t1.l;
    //     }
    // }

    // fast input reader class;
    
}
