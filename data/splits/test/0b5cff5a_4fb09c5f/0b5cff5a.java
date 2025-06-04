import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

public class E {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);


        int t= in.nextInt();
        while(t-->0) {

            int n = in.nextInt();
            int k = in.nextInt();
            int[] a= in.readArray(k);
            int[] temp = in.readArray(k);
            int[] pre = new int[n];
            Arrays.fill(pre,Integer.MAX_VALUE);
            int[] suf = new int[n];
            Arrays.fill(suf,Integer.MAX_VALUE);
            for(int i = 0;i<k;i++){
                pre[a[i]-1]=temp[i];
                suf[a[i]-1]=temp[i];
            }
            int min = Integer.MAX_VALUE;
            Arrays.sort(a);
            for(int i=a[0]-1;i<n;i++){
                min = Math.min(min,pre[i]);
                pre[i] = min;
                min++;

            }
            min = Integer.MAX_VALUE;
            for(int i = a[k-1]-1;i>=0;i--){
                min = Math.min(min,suf[i]);
                suf[i] = min;
                min++;
            }

            for(int i=0;i<n;i++)
                out.print(Math.min(pre[i],suf[i])+" ");
            out.println();
        }
        out.close();
    }




    
    
    

    

    
    
    

    
}
