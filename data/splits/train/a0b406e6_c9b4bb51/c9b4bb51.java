import java.lang.*;
import java.util.*;
import java.io.*;
import java.math.*;

public class Main {
    static void deal(int n,int[] arr) {
        HashSet<Integer> hs = new HashSet<>();
        int max = (int)Math.pow(3,n);
        int[] sum = new int[max];
        hs.add(0);
        for(int i=1;i<max;i++) {
            int s = 0;
            int num = i;
            int index = 0;
            while(num>0) {
                int d = num % 3;
                num = num /3;
                if(d==1) {
                    s+=arr[index];
                } else if(d==2) {
                    s-=arr[index];
                }
                index++;
            }
            sum[i] = s;
            if(s==0) {
                out.println("YES");
                return;
            }
            hs.add(sum[i]);
        }
        out.println("NO");
    }

    

	public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
        int t = sc.nextInt();
        for(int i=0;i<t;i++) {
            int n = sc.nextInt();
            int[] arr = new int[n];
            for(int j=0;j<n;j++) {
                arr[j] = sc.nextInt();
            }
            deal(n,arr);
        }
        out.close();
    }
    
    //-----------PrintWriter for faster output---------------------------------
    public static PrintWriter out;
    
    //-----------MyScanner class for faster input----------
    
}
