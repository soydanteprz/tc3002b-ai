import java.util.*;

public class contestA {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int t = scanner.nextInt();
        int inf = 1000300300;
        while (t-->0){
            int n = scanner.nextInt();
            int k = scanner.nextInt();
            int[] c = new int[n];
            Arrays.fill(c,inf);
            int[] a = new int[k];
            int[] b = new int[k];
            for(int i=0;i<k;++i) a[i] = scanner.nextInt() - 1;
            for(int i=0;i<k;++i) b[i] = scanner.nextInt();
            for(int i=0;i<k;++i) c[a[i]] = b[i];
            for(int i=1;i<n;++i){
                c[i] = Math.min(c[i],c[i-1]+1);
            }
            for(int i=n-2;i>=0;--i){
                c[i] = Math.min(c[i],c[i+1]+1);
            }
            for(int i=0;i<n;++i) System.out.print(c[i]+" ");
            System.out.println();
        }
    }
}
