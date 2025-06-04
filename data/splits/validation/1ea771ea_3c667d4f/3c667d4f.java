import java.util.*;
public class j
{
  public static void main(String args[])
  {
    Scanner in=new Scanner(System.in);
    int n=in.nextInt();
    while(n-->0)
    {
      int len=in.nextInt();
      int t=in.nextInt();
      int pos[]=new int[t];
      int temp[]=new int[t];
      for(int i=0;i<t;i++)
      pos[i]=in.nextInt();
      for(int i=0;i<t;i++)
      temp[i]=in.nextInt();
      long range[]=new long[len];
      Arrays.fill(range,Long.MAX_VALUE-10000);
      for(int i=0;i<t;i++)
      range[pos[i]-1]=temp[i];
      for(int i=1;i<len;i++)
      {
        range[i]=Math.min(range[i],1+range[i-1]);
      }
      for(int i=len-2;i>=0;i--)
      {
      range[i]=Math.min(range[i+1]+1,range[i]);
      }
      for(int i=0;i<len;i++)
      {
        System.out.print(range[i]+" ");
      }System.out.println();
    }
  }
}
