import java.io.*;
import java.util.*;

public class Codeforce {

    

    static int mod = (int) (1e9 + 7);

    

    

    

    

    

    

    public static void main(String arg[]) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int t = Integer.parseInt(br.readLine());
        StringBuilder sb = new StringBuilder();
        while (t-- > 0) {
                                br.readLine();
                        String s[]=br.readLine().split(" ");
                        int n=Integer.parseInt(s[0]);
                        int k=Integer.parseInt(s[1]);
                        long a[]=new long[n];
                        long b[]=new long[n];
                        long c[]=new long[n];
                        Arrays.fill(c,Integer.MAX_VALUE);
                        String s1[]=br.readLine().split(" ");
                        s=br.readLine().split(" ");
                        for(int i=0;i<k;i++)
                            c[Integer.parseInt(s1[i])-1]=Integer.parseInt(s[i]);

                        long p=Integer.MAX_VALUE;
                        for(int i=0;i<n;i++)
                        {
                            p=Math.min(p+1,c[i]);
                            a[i]=p;
                        }
                        p=Integer.MAX_VALUE;
                        for(int i=n-1;i>=0;i--){
                            p=Math.min(p+1,c[i]);
                            b[i]=p;
                        }
                        for(int i=0;i<n;i++)
                            sb.append((Math.min(a[i],b[i]))+" ");
                        sb.append("\n");

                       }
        System.out.println(sb);
    }
}
