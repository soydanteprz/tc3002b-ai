import java.io.*;
import java.util.*;
import java.text.DecimalFormat;

public class C {
    static long mod=(long)1e9+7;
    static long mod1=998244353;
    static InputStream inputStream = System.in;
    static OutputStream outputStream = System.out;
    static InputReader in = new InputReader(inputStream);
    static PrintWriter out = new PrintWriter(outputStream);

    public static void main(String[] args) throws IOException {

        int t= in.nextInt();
        while(t-->0) {

            int n = in.nextInt();
            ArrayList<Pair>[] adj = new ArrayList[5];
            for(int i = 0;i<5;i++) adj[i] = new ArrayList<>();
            for(int i = 0;i<n;i++){
                char[] arr = in.next().toCharArray();
                int[] count = new int[5];
                for(char c:arr)
                    count[c-'a']++;
                for(int j = 0;j<5;j++){
                    adj[j].add(new Pair(count[j],arr.length-count[j]));
                }
            }
            int max = 0;
            for(int i = 0;i<5;i++){
                Collections.sort(adj[i]);
                int count = 0;
                int sum1 = 0;
                int sum2 = 0;
                for(Pair p:adj[i]){
                    if(sum1+p.a>sum2+p.b){
                        count++;
                        sum1+=p.a;
                        sum2+=p.b;
                    }
                }
                max = Math.max(max,count);
            }

            out.println(max);

        }
        out.close();
    }

    static class Pair implements Comparable<Pair>{
        int a,b;
        Pair(int a, int b){
            this.a = a;
            this.b = b;
        }

        
    }


    static final Random random=new Random();

    
    
    
    

    

    
    
    

    
}
