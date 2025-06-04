import java.io.*;

public class Main {
    static BufferedReader BR = new BufferedReader(new InputStreamReader(System.in));
    static StreamTokenizer IN = new StreamTokenizer(BR);
    static PrintWriter OUT = new PrintWriter(new BufferedOutputStream(System.out));
    static int nextInt() throws IOException {
        IN.nextToken();
        return (int) IN.nval;
    }
    
    
    


    
    
    public static void main(String[] args) throws IOException {
//        Scanner sc = new Scanner(System.in);
        int T = nextInt();

        while(T-- > 0) {
            int N = nextInt();

            int[] a = new int[N];

            for (int i = 0; i < N; i++) {
                a[i] = nextInt();
            }
            if (N % 2 == 0) {
               for (int i = 0; i < N; i+=2) {
                   OUT.print(a[i+1] + " ");
                   OUT.print(-1 * a[i] + " ");
               }
            } else{
                for (int i = 0; i < N-3; i+=2) {
                    OUT.print(a[i+1] + " ");
                    OUT.print(-1 * a[i] + " ");
                }
                if(a[N-3] + a[N-2] != 0) {
                    OUT.print(-a[N-1] + " " + -a[N-1] + " " + (a[N-3] + a[N-2]));
                } else if(a[N-2] + a[N-1] != 0) {
                    OUT.print((a[N-2] + a[N-1]) + " " + -a[N-3] + " " + -a[N-3]);
                } else {
                    OUT.print(-a[N-2] + " " + (a[N-1] + a[N-3]) + " " + -a[N-2]);
                }
            }

            OUT.println();
        }

        OUT.flush();
    }
}
